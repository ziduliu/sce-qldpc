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
CONCEPT = ("For F8-additive semidirect C14, order 112, shapes (1,2)x(2,5), full-generating bidirectional-Sidon cells and independently challenged logical witnesses should improve five-code fitness above 109.4042 in cell 0.250. Non-improvement refutes this experiment; sampled distance readings remain provisional upper bounds.")
GROUP = ("Jump from perfect SL2(F5), order 120, to the solvable affine group F8-additive semidirect C14, order 112. A primitive F8 multiplier has order seven; the complement's seventh power is central. This group has centre two and 16 conjugacy classes. The measured order-112, centre-two, 16-class beds reached 343.4 on (2,3)x(2,3) and 331.7 on (3,2)x(3,2); compatible invariants motivate an affine experiment, not fingerprint identification or distance transfer. The present shape bed is absent from BEDS.")
SHAPES = ("Keep A=(1,2), B=(2,5): n=12*112=1344, k>=3*112=336 and guaranteed rate 0.250; this shape permits q<=124. Unequal components have 224 variables/112 checks and 560 variables/224 checks, nominal rates 1/2 and 3/5. Parent measured minima 20..22 were entirely sector one: X usually followed Acol1 across B columns, while Z used paired A-column strips at Bcol2/3/4. Challenge both corresponding kernels and gate actual full ranks to [0.225,0.275).")
FILLING = ("Twenty cards cross four B incidence profiles with five quotient-support recipes. A counts remain (4,4), B rows total eight and columns have degrees three/four, so both CSS row caps remain twelve. Binomial ratios have order seven or fourteen; larger cells relatively generate all 112 elements and have distinct directed differences in both orientations. Accumulate genuine component witnesses, redraw implicated cells independently and independently deepen finalists without erasing earlier readings. Compare exact full k and original-stabilizer-span-verified logical embeddings, then verify native components, weights and CSS. Neither component searches nor sampled logical readings certify distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_F8add_C14_12x25_sidon_quotient_witness_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F8-additive semidirect C14 requires order 112")
    # F8 = F2[x]/(x^3+x+1), encoded by three coefficient bits.
    # (a,t) represents translation a followed by multiplier x^t.
    # The C14 coordinate is retained: (0,7) is a central involution.
    def field_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return result

    products = [[field_mul(a, b) for b in range(8)]
                for a in range(8)]
    powers = [1]
    for _ in range(6):
        powers.append(products[powers[-1]][2])
    assert len(set(powers)) == 7 and products[powers[-1]][2] == 1
    els = [(a,t) for t in range(14) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 14)

    def mul(g, h):
        a,t = red(g)
        b,u = red(h)
        return (a ^ products[powers[t % 7]][b], (t+u) % 14)

    def inv(g):
        a,t = red(g)
        ti = (-t) % 14
        return (products[powers[ti % 7]][a], ti)

    def idx(g):
        a,t = red(g)
        return 8*t+a

    return dict(elements=els, ident=(0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty affine quotient/incidence cards with witnessed repair."""
    q = G.q
    assert q == 112
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    topology, model = divmod(idx % 20, 5)
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

    # Quotient relations change independently between cells; no slot
    # is constructed by applying a common automorphism to another slot.
    triple_profiles = (
        (0,1,2), (0,1,3), (0,2,5), (0,1,4), (0,2,3),
    )
    quartet_profiles = (
        (0,1,2,3), (0,1,2,4), (0,1,3,4),
        (0,1,2,5), (0,1,3,5),
    )
    bin_orders = ((14,), (7,), (7,14), (14,), (7,14))
    ratios = {order: [g for g in range(q) if orders[g] == order]
              for order in (7, 14)}
    span_cache = {}

    def generating_sidon(cell):
        """Native subgroup and difference tests, not a rewritten lift."""
        left, right = set(), set()
        for g in cell:
            for h in cell:
                if g != h:
                    left.add(table[inverses[g]][h])
                    right.add(table[g][inverses[h]])
        expected = len(cell)*(len(cell)-1)
        if len(left) != expected or len(right) != expected:
            return False
        roots = tuple(sorted(table[inverses[cell[0]]][g]
                             for g in cell[1:]))
        if roots not in span_cache:
            reached, pending = {ident}, [ident]
            for value in pending:
                for root in roots:
                    image = table[value][root]
                    if image not in reached:
                        reached.add(image)
                        pending.append(image)
            span_cache[roots] = len(reached) == q
        return span_cache[roots]

    def fresh_cell(count, policy):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            root = rng.randrange(q)
            order = rng.choice(bin_orders[policy])
            ratio = rng.choice(ratios[order])
            ids = [root, table[root][ratio]]
            assert ids[0] != ids[1]
            assert relative_order[ids[0]][ids[1]] == order
            return [G.elements[g] for g in ids]

        assert count in (3, 4)
        profile = (triple_profiles if count == 3
                   else quartet_profiles)[policy]
        for _ in range(4096):
            phase = rng.randrange(14)
            cell = [(rng.randrange(8), (phase+t) % 14)
                    for t in profile]
            ids = [G.idx(g) for g in cell]
            if generating_sidon(ids):
                assert len(set(ids)) == count
                return cell
        raise ValueError("generating affine quotient-Sidon recipe exhausted")

    def support_b():
        # Compare the 2+2 and 1+3 degree-four split, with and without
        # a concentrated degree-three column. Counts are prescriptions,
        # not transposed or relabeled archived fillings.
        profiles = (
            ((2,2,1,1,2), (1,1,2,2,2)),
            ((1,2,1,2,2), (3,1,2,1,1)),
            ((2,3,1,1,1), (2,0,2,2,2)),
            ((1,3,2,1,1), (3,0,1,2,2)),
        )
        grid = [list(row) for row in profiles[topology]]
        assert [sum(row) for row in grid] == [8, 8]
        assert sorted(sum(row[j] for row in grid) for j in range(5)) \
            == [3,3,3,3,4]
        assert _grid_weights([[4, 4]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[4, 4]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def verified_component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        # Empty dummy sides isolate components in the verified helper.
        # These dummy constructions are never submitted as codes.
        if side == "A":
            rows, _, _ = lifted_matrices(probe, indexed, [[[]]])
            width = 2*q
            assert len(rows) == q
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], indexed)
            width = 5*q
            assert len(rows) == 2*q
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

    def kernel_probe(rows, columns, trials, diagnostic, previous=()):
        """Accumulate actual null words; high readings certify no distance."""
        width = len(columns)
        order = list(range(width))
        best, witness = width+1, None
        retained = sorted(set(previous),
                          key=lambda w: (w.bit_count(), w))[:24]
        found = {}

        def observe(word):
            nonlocal best, witness
            if not word:
                return
            weight = word.bit_count()
            if weight < best:
                best, witness = weight, word
            if weight <= best+4:
                found[word] = weight
                if len(found) > 96:
                    keep = sorted(found, key=lambda w: (found[w], w))[:48]
                    found.clear()
                    found.update((w, w.bit_count()) for w in keep)

        for word in retained:
            assert all((word & row).bit_count() % 2 == 0 for row in rows)
            observe(word)

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    observe(word)

            light = sorted(set(basis+retained),
                           key=lambda w: (w.bit_count(), w))[:20]
            reducers = basis+retained
            for word in light[:16]:
                for other in reducers:
                    observe(word ^ other)

            diagnostic.shuffle(reducers)
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in reducers:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
            if len(light) >= 3:
                for _ in range(64):
                    a, b, c = diagnostic.sample(light, 3)
                    observe(a ^ b ^ c)
            retained = sorted(
                (w for w in found if w.bit_count() <= best+4),
                key=lambda w: (w.bit_count(), w))[:24]
            if best <= 4:
                break

        if witness is None:
            raise ValueError("no component kernel witness")
        answer = sorted(set(retained) | {witness},
                        key=lambda w: (w.bit_count(), w))[:24]
        assert all(all((word & row).bit_count() % 2 == 0
                       for row in rows) for word in answer)
        return best, witness, answer

    def merit(record):
        return (tuple(-x for x in record["small"]),
                record["d"],
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        policy = (model + side_id + cohort) % 5
        support = [[4, 4]] if side == "A" else support_b()
        seed = diagnostic_seed + 100003*side_id
        records, signatures, bank = [], set(), set()

        def survives(record, word):
            value = 0
            for c in bits(word):
                value ^= record["columns"][c]
            return value == 0

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
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "cycles": stats[1], "trial": trial,
                "rank": gf2_rank(rows), "d": len(columns)+1,
                "words": [], "word": None,
            }

        def refresh(record):
            # A historical word is relevant only if its CURRENT syndrome
            # vanishes. Earlier discoveries for this matrix never vanish.
            words = list(record["words"])
            words.extend(word for word in bank
                         if word.bit_count() <= record["d"]+4
                         and survives(record, word))
            if words:
                record["words"] = sorted(set(words),
                    key=lambda w: (w.bit_count(), w))[:24]
                record["word"] = record["words"][0]
                record["d"] = min(record["d"],
                                   record["word"].bit_count())

        def score(record):
            refresh(record)
            return merit(record)

        def diagnose(record, trials, turn):
            refresh(record)
            reading, word, words = kernel_probe(
                record["rows"], record["columns"], trials,
                random.Random(seed+700001*turn), record["words"])
            record["d"] = min(record["d"], reading)
            record["words"] = sorted(
                set(record["words"]+words),
                key=lambda w: (w.bit_count(), w))[:24]
            record["word"] = record["words"][0]
            bank.update(words[:6])
            if len(bank) > 96:
                keep = sorted(bank,
                    key=lambda w: (w.bit_count(), w))[:72]
                bank.clear()
                bank.update(keep)
            refresh(record)
            assert survives(record, word)

        for trial in range(24):
            grid = [[fresh_cell(count, policy) for count in row]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], r["cycles"], -r["rank"], r["trial"]))
        selected = records[:4]
        if not selected:
            raise ValueError("empty affine quotient component shortlist")
        for number, record in enumerate(selected):
            diagnose(record, 6, 1+number)
        current = max(selected, key=score)

        occupied = [(i, j) for i, row in enumerate(support)
                    for j, count in enumerate(row) if count]
        for step in range(8):
            refresh(current)
            active = [c//q for word in current["words"][:4]
                      for c in bits(word)]
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            changed = set()
            for move in range(2 if step % 4 == 3 else 1):
                j = (rng.choice(active) if active
                     else rng.randrange(len(support[0])))
                targeted = [p for p in occupied
                            if p[1] == j and p not in changed]
                available = [p for p in occupied if p not in changed]
                choices = (targeted if targeted and rng.random() < 0.85
                           else available)
                dense = [p for p in choices if support[p[0]][p[1]] >= 2]
                if dense and rng.random() < 0.8:
                    choices = dense
                i, j = rng.choice(choices)
                changed.add((i, j))
                # Independent replacements, not a common cell gauge.
                grid[i][j] = fresh_cell(support[i][j], policy)
            proposed = inspect(grid, 100+step)
            if proposed is None or proposed["small"] > current["small"]:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+2*q:
                continue
            diagnose(proposed, 4, 20+step)
            selected.append(proposed)
            pk, ck = score(proposed), score(current)
            if pk >= ck or (pk[:2] == ck[:2] and rng.random() < 0.10):
                current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for number, record in enumerate(finalists):
            diagnose(record, 16, 101+number)
        finalists.sort(key=score, reverse=True)
        for number, record in enumerate(finalists[:2]):
            # Independent deeper diagnostics can only lower the reading.
            diagnose(record, 32+4*(cohort % 3), 211+number)
        finalists.sort(key=score, reverse=True)
        return finalists[:2]

    aa, bb = component_pool("A"), component_pool("B")
    mask = (1 << q)-1

    def span(rows):
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

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(a, b, hx, hz, n):
        """Sample actual logicals modulo ORIGINAL stabilizer row spans.

        n+1 denotes no detected embedding, not an exclusion certificate.
        """
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            diagnostic = random.Random(
                diagnostic_seed + (3100009 if kind == "Z" else 4100011))
            candidates = set()
            for word in record["words"][:2]:
                for location in range(5 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j+location)*q) for j in range(2))
                    else:
                        actual = word << (location*5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            for actual in sorted(candidates,
                                 key=lambda w: (w.bit_count(), w))[:6]:
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
                neighborhood = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:24]
                if len(neighborhood) >= 2:
                    for _ in range(64):
                        first, second = diagnostic.sample(neighborhood, 2)
                        proposed = actual ^ first ^ second
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    best = None
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 12*q == 1344
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 3*q and 9*n <= 40*k < 11*n):
                continue
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert max(row.bit_count() for row in hx+hz) <= 12

            actual_a = [
                sum(((hx[r] >> (5*j*q)) & mask) << (j*q)
                    for j in range(2)) for r in range(q)]
            actual_b = [row & ((1 << (5*q))-1)
                        for row in hz[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]

            logicals = logical_challenge(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (
                tuple(-(x+y) for x, y in zip(a["small"], b["small"])),
                k*detected*detected, detected, a["d"]+b["d"],
                -a["cycles"][0]-b["cycles"][0])
            if best is None or key > best[0]:
                best = (key, a, b, hx, hz, k, logicals)

    if best is None:
        raise ValueError("affine quotient pairings missed cell 0.250")
    key, a, b, hx, hz, k, logicals = best
    assert css_ok(hx, hz)
    return {
        "idx": idx, "cells": {"A": a["grid"], "B": b["grid"]}, "k": k,
        "card": (topology, model, cohort),
        "component_readings": (a["d"], b["d"]),
        "component_small_kernels": (a["small"], b["small"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "verified_localized_logical_readings": logicals,
        "selection_quality": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the jointly designed and screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings remain unchanged under longer measurement on the SL2(F5) bed; measured slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
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
