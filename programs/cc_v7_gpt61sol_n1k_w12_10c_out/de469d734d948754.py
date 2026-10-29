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
CONCEPT = ("Hypothesis: for SL2(F5), order 120, A=(2,1), B=(5,2), localized cokernel protection followed by surviving-witness repair and original-span quantum-logical reduction raises five-code fitness above 109.4042 in cell 0.250. Transposition alone predicts no distance gain; search readings remain provisional upper bounds.")
GROUP = ("Keep the exact SL2(F5) matrix-group construction and order 5*(25-1)=120, without parameter changes. Its wide bed leads cell 0.250 at 109.4042; the tall (2,1)x(5,2) bed measured 95.8240. Those measurements motivate new fillings and stronger localized diagnostics, not an orientation advantage or certified distance transfer.")
SHAPES = ("Change to A=(2,1), B=(5,2), retaining q=120, n=12*q=1440 and k>=3*q=360, guaranteed rate 0.250; the unrestricted shape permits q<=124. Raw components are tall, while their relevant transpose components have 240 variables/120 checks and 600 variables/240 checks, nominal rates 1/2 and 3/5. Parent measured minima 20..22 occupied sector-1 Acol/Bcol strips, especially Acol1 and Bcol2/3/4; corresponding tall families occupy sector 2. Exact full ranks retain [0.225,0.275).")
FILLING = ("Twenty cards cross four explicit tall B incidence profiles with five binomial relative-order prescriptions. A counts are (4,4); B column totals are eight and row totals are either (4,3,3,3,3) or (4,4,3,3,2), giving HX/HZ maxima twelve. Larger cells must have distinct directed differences in both orientations and full-group relative generation. Verified unit probes supply transpose components. Localized kernels are exactly enumerated when nullity<=14 and heuristically searched otherwise; surviving witnesses direct cell repair and independent finalist challenges. Final pairings test actual nontrivial sector-2 logical embeddings modulo original stabilizer spans and reduce their weights. Full ranks, weights and CSS gate submission; no component heuristic or budgeted search certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_SL2F5_21x52_sidon_localized_quantumrepair_v3"

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

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (5, 2)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty tall incidence/order cards with localized witnessed repair."""
    q = G.q
    assert q == 120
    assert A_SHAPE == (2, 1) and B_SHAPE == (5, 2)
    topology = (idx % 20) // 5
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

    span_cache = {}

    def admissible(cell):
        rels = [relative_order[g][h]
                for i, g in enumerate(cell) for h in cell[:i]]
        if 2 in rels:
            return None
        if len(cell) >= 3:
            left, right = set(), set()
            for g in cell:
                for h in cell:
                    if g != h:
                        left.add(table[inverses[g]][h])
                        right.add(table[g][inverses[h]])
            wanted = len(cell)*(len(cell)-1)
            if len(left) != wanted or len(right) != wanted:
                return None
            roots = tuple(sorted(
                table[inverses[cell[0]]][g] for g in cell[1:]))
            if roots not in span_cache:
                reached, queue = {ident}, [ident]
                for value in queue:
                    for root in roots:
                        product = table[value][root]
                        if product not in reached:
                            reached.add(product)
                            queue.append(product)
                span_cache[roots] = len(reached) == q
            if not span_cache[roots]:
                return None
        return rels

    def fresh_cell(count, policy):
        if not count:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            prescriptions = ((5,), (6,), (10,), (4, 10), (5, 6))
            wanted = rng.choice(prescriptions[policy])
            ratios = [g for g in range(q) if orders[g] == wanted]
            root = rng.randrange(q)
            cell = [root, table[root][rng.choice(ratios)]]
            assert relative_order[cell[0]][cell[1]] == wanted
            return [G.elements[g] for g in cell]

        best, best_cost = None, None
        for attempt in range(512):
            cell = rng.sample(range(q), count)
            rels = admissible(cell)
            if rels is None:
                continue
            cost = sum(penalties[policy].get(order, 30)
                       for order in rels)
            if best_cost is None or cost < best_cost:
                best, best_cost = cell, cost
            if best is not None and (best_cost == 0 or attempt >= 23):
                break
        if best is None:
            raise ValueError("whole-Sidon generating-cell recipe exhausted")
        return [G.elements[g] for g in best]

    def support_b():
        # Each column totals eight. The degree-two row in cards 2/3
        # is split 1+1, never an isolated binomial kernel block.
        profiles = (
            ((2,2), (1,2), (2,1), (1,2), (2,1)),
            ((1,3), (1,2), (2,1), (2,1), (2,1)),
            ((2,2), (2,2), (1,2), (2,1), (1,1)),
            ((1,3), (3,1), (2,1), (1,2), (1,1)),
        )
        grid = [list(row) for row in profiles[topology]]
        assert [sum(row[j] for row in grid) for j in range(2)] == [8, 8]
        assert _grid_weights([[4], [4]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[4], [4]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def verified_component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        # Unit partners expose the authoritative transpose convention
        # directly in sector two. They are screening-only constructions.
        unit = [[[ident]]]
        if side == "A":
            _, rows, _ = lifted_matrices(probe, indexed, unit)
            width = 2*q
            assert len(rows) == q
            rows = [row >> q for row in rows]
        else:
            rows, _, _ = lifted_matrices(probe, unit, indexed)
            width = 5*q
            assert len(rows) == 2*q
            rows = [row >> (2*q) for row in rows]
        assert all(row >> width == 0 for row in rows)
        columns = [0]*width
        for check, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << check
        return rows, columns

    def quality(rows, columns):
        """Exact kernel-3/4 and four-cycle counts; six-cycle proxy."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return None
        pairs = {}
        three = four_words = 0
        for i, a in enumerate(columns):
            for b in columns[:i]:
                syndrome = a ^ b
                three += int(syndrome in singles)
                previous = pairs.get(syndrome, 0)
                four_words += previous
                pairs[syndrome] = previous+1

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

        def observe(word):
            nonlocal best, witness
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word

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
        return best, witness

    local_cache = {}

    def localized_word(columns):
        """Exact minimum at nullity <=14; verified upper witnesses otherwise."""
        key = tuple(columns)
        if key in local_cache:
            return local_cache[key]
        pivots, basis = {}, []
        for c, source in enumerate(columns):
            value, word = source, 1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        best = None

        def note(word):
            nonlocal best
            if word and (best is None or
                         word.bit_count() < best.bit_count()):
                best = word

        if len(basis) <= 14:
            word = 0
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length()-1]
                note(word)
        else:
            for i, word in enumerate(basis):
                note(word)
                for other in basis[:i]:
                    note(word ^ other)
        if best is not None:
            value = 0
            for c in bits(best):
                value ^= columns[c]
            assert value == 0
        local_cache[key] = best
        return best

    def merit(record):
        return (tuple(-x for x in record["small"]),
                record["d"],
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        policy = (model + side_id + cohort) % 5
        support = [[4], [4]] if side == "A" else support_b()
        seed = diagnostic_seed + 100003*side_id
        records, signatures, bank = [], set(), set()

        def survives(record, word):
            syndrome = 0
            for c in bits(word):
                syndrome ^= record["columns"][c]
            return syndrome == 0

        def refresh(record):
            for word in bank:
                if word.bit_count() < record["d"] and survives(record, word):
                    record["d"] = word.bit_count()
                    record["words"].append(word)
            record["words"] = sorted(
                set(record["words"]), key=lambda w: (w.bit_count(), w))[:8]

        def score(record):
            refresh(record)
            return merit(record)

        def inspect(grid, trial):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return None
            signatures.add(signature)
            rows, columns = verified_component(grid, side)
            stats = quality(rows, columns)
            if stats is None:
                return None
            local_words = []
            for block in range(len(grid)):
                word = localized_word(columns[block*q:(block+1)*q])
                if word is not None:
                    local_words.append(word << (block*q))
            reading = min((w.bit_count() for w in local_words),
                          default=len(columns)+1)
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "cycles": stats[1], "trial": trial,
                "localized": reading, "local_words": local_words,
                "d": reading, "words": list(local_words),
            }

        def diagnose(record, trials, offset):
            diagnostic = random.Random(seed+offset)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            record["d"] = min(record["d"], reading)
            record["words"].append(word)
            bank.add(word)
            bank.update(record["local_words"])
            if len(bank) > 128:
                keep = sorted(bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(keep)
            if "rank" not in record:
                record["rank"] = gf2_rank(record["rows"])
            refresh(record)
            assert all(survives(record, w) for w in record["words"])

        for trial in range(64 if side == "A" else 32):
            grid = [[fresh_cell(count, policy) for count in row]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)

        # A localized witness is an actual obstruction, not just a
        # cycle statistic. Do not let attractive girth hide that word.
        records.sort(key=lambda r: (
            r["small"], -r["localized"], r["cycles"], r["trial"]))
        selected = records[:6]
        if not selected:
            raise ValueError("empty localized tall-component shortlist")
        for record in selected:
            diagnose(record, 8, 17)
        selected.sort(key=score, reverse=True)
        current = selected[0]

        for step in range(16):
            refresh(current)
            word = rng.choice(current["words"][:4])
            active = {c//q for c in bits(word)}
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            positions = [(i, j) for i, row in enumerate(grid)
                         for j, cell in enumerate(row) if cell]
            changed = set()
            for move in range(2 if step % 4 == 3 else 1):
                available = [p for p in positions if p not in changed]
                targeted = [p for p in available if p[0] in active]
                choices = (targeted if targeted and rng.random() < 0.85
                           else available)
                dense = [p for p in choices if len(grid[p[0]][p[1]]) >= 2]
                if dense:
                    choices = dense
                if not choices:
                    break
                i, j = rng.choice(choices)
                changed.add((i, j))
                grid[i][j] = fresh_cell(len(grid[i][j]), policy)

            proposed = inspect(grid, 100+step)
            if proposed is None or proposed["small"] > current["small"]:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+2*q:
                continue
            diagnose(proposed, 6, 300007+7919*step)
            selected.append(proposed)
            pk, ck = score(proposed), score(current)
            if pk >= ck or (
                    pk[:2] == ck[:2] and rng.random() < 0.12):
                current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            # Two fresh diagnostics only tighten this actual component.
            diagnose(record, 40+4*(cohort % 3), 700001)
            diagnose(record, 16, 1900003)
        finalists.sort(key=score, reverse=True)
        return finalists[:2]

    def span_pivots(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, pivots):
        while word:
            lead = word.bit_length()-1
            old = pivots.get(lead)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(hx, hz, a, b, n):
        """Actual sector-two logicals, checked against ORIGINAL spans."""
        mask = (1 << q)-1
        n1 = 2*q
        readings = []
        for kind, opposite, stabilizers, record in (
                ("X", hz, hx, a), ("Z", hx, hz, b)):
            pivots = span_pivots(stabilizers)
            diagnostic = random.Random(
                diagnostic_seed + (3100009 if kind == "X" else 4100011))
            candidates = []
            for word in record["words"][:6]:
                for fixed in range(5 if kind == "X" else 2):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << (n1+(5*i+fixed)*q)
                            for i in range(2))
                    else:
                        actual = word << (n1+5*fixed*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if nontrivial(actual, pivots):
                        candidates.append(actual)

            found = n+1
            for actual in sorted(set(candidates), key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:24]
                for _ in range(160):
                    first, second = diagnostic.sample(nearby, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and nontrivial(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                found = min(found, actual.bit_count())
            readings.append(found)
        return tuple(readings)

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]-ab[1]["cycles"][0]),
        reverse=True)

    best = None
    mask = (1 << q)-1
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 12*q == 1440
        assert len(hx) == 4*q and len(hz) == 5*q
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 3*q and 9*n <= 40*k < 11*n):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)

        # Independently confirm the unit-probe transpose conventions
        # against the FINAL construction and its actual sector order.
        actual_a = [
            sum(((hz[t] >> ((2+5*i)*q)) & mask) << (i*q)
                for i in range(2))
            for t in range(q)
        ]
        actual_b = [
            (hx[t] >> (2*q)) & ((1 << (5*q))-1)
            for t in range(2*q)
        ]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        logicals = logical_readings(hx, hz, a, b, n)
        detected = min(a["d"], b["d"], *logicals)
        key = (
            tuple(-(x+y) for x, y in zip(a["small"], b["small"])),
            detected, k*detected*detected,
            a["d"]+b["d"], -a["cycles"][0]-b["cycles"][0],
        )
        if best is None or key > best[0]:
            best = (key, a, b, k, logicals)
    if best is None:
        raise ValueError("localized tall pairings missed the target-rate gate")
    key, a, b, k, logicals = best
    return {
        "idx": idx, "cells": {"A": a["grid"], "B": b["grid"]}, "k": k,
        "card": (topology, model, cohort),
        "orientation": "tall-authoritative-transpose",
        "component_readings": (a["d"], b["d"]),
        "localized_component_readings": (a["localized"], b["localized"]),
        "component_small_kernels": (a["small"], b["small"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "verified_logical_readings": tuple(
            None if d > 1440 else d for d in logicals),
        "quality": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the jointly designed and screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable short-budget readings remain unchanged under longer search on this SL2(F5) bed; measured parent slots 6, 28 and 34 fell from 24 to 22, 22 to 20 and 23 to 20.
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
