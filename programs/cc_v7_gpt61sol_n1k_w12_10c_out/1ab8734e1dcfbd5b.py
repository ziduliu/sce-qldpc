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
CONCEPT = ("For F9-additive semidirect C4, order 36, A=(2,5), B=(3,7), multirow degree-three A and degree-three/four B components should improve five-code fitness above 73.143 in cell 0.300. Non-improvement refutes this prediction; search readings remain provisional upper bounds.")
GROUP = ("Retain the quadratic-field additive-by-cyclic semidirect construction, resizing the characteristic from five to three and the complement from eight to four. In F3[u]/(u^2-2), multiplication by u has exact order four and acts irreducibly. Order 36 gives n=1476. The cyclic order-35 bed on these shapes reached 49.5; irreducible nonabelian mixing is an experiment, not a distance-transfer theorem.")
SHAPES = ("Change A to (2,5) and B to (3,7): n=41*36=1476, k>=12*36=432 and guaranteed rate 12/41=0.292683 target cell 0.300; q_max=36. Components have 180 variables/72 checks and 252 variables/108 checks, nominal rates 3/5 and 4/7. Parent measured X minima were weight-sixteen sector-1 single-Acol strips across three Bcols. Both new components have multiple check-block rows. Every submitted pair must pass the exact [0.275,0.325) rate gate.")
FILLING = ("Twenty cards cross four support-concentration prescriptions with five B degree profiles. A column degrees are three and row totals seven/eight. B column degrees are three/four and balanced row totals are at most nine, giving HX<=8+4=12 and HZ<=9+3=12. Nonuniform supports, zero blocks and independently searched relative labels replace five trinomials. Authoritative monomial blocks supply all component matrices. Exact small-word collisions, genuine kernel witnesses and cycle counts guide repairs; independent finalist probes only lower readings. Original stabilizer spans verify and reduce localized logical candidates. Full ranks, assembly agreement, weights and CSS gate submissions; no heuristic certifies distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r030_F9add_C4_25x37_multirow_degree34_logicalrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F9-additive semidirect C4 requires order 36")
    # Same quadratic-field additive-by-cyclic semidirect rule.
    # F9 = F3[u]/(u^2-2); u^2=-1 and u has exact order four.
    # (a,b,t) acts as z -> u^t*z + (a+b*u).
    powers = ((1,0), (0,1), (2,0), (0,2))
    els = [(a,b,t) for t in range(4)
           for b in range(3) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 3, g[2] % 4)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = powers[t]
        return ((a+v*c+2*w*d) % 3,
                (b+v*d+w*c) % 3, (t+u) % 4)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % 4
        v,w = powers[ti]
        return ((-v*a-2*w*b) % 3,
                (-v*b-w*a) % 3, ti)

    def idx(g):
        a,b,t = red(g)
        return 9*t+3*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _legacy_affine_ctx(rng, G, idx):
    """Unused historical single-row optimizer; NEW_CTX below replaces it."""
    q = G.q
    assert q == 200
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

    # These specify occupancies, not automorphism images of one filling.
    # Every profile includes an odd quotient difference. No profile
    # contains a difference of four, which would force an involution.
    repeated = (0, 0, 1)
    consecutive = (0, 1, 2)
    asymmetric = (0, 1, 3)
    a_plans = (
        (repeated, repeated),
        (repeated, consecutive),
        (consecutive, consecutive),
        (repeated, asymmetric),
        (consecutive, asymmetric),
    )
    b_plans = (
        (repeated, repeated, consecutive),
        (repeated, consecutive, asymmetric),
        (consecutive, consecutive, asymmetric),
        (asymmetric, asymmetric, asymmetric),
    )
    plans = {"A": a_plans[model], "B": b_plans[topology]}
    span_cache = {}

    def spans_group(cell):
        roots = tuple(sorted(
            table[inverses[cell[0]]][g] for g in cell[1:]))
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

    def fresh_cell(count, profile):
        assert count == 3
        for _ in range(512):
            phase = rng.randrange(8)
            cell = []
            for t in profile:
                g = (rng.randrange(5), rng.randrange(5),
                     (t+phase) % 8)
                while g in cell:
                    g = (rng.randrange(5), rng.randrange(5),
                         (t+phase) % 8)
                cell.append(g)
            ids = [G.idx(g) for g in cell]
            if any(relative_order[g][h] == 2
                   for i, g in enumerate(ids) for h in ids[:i]):
                continue
            if spans_group(ids):
                return cell
        raise ValueError("full-generating affine trinomial recipe exhausted")

    def support_b():
        grid = [[3, 3, 3]]
        assert _grid_weights([[3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    # Obtain both monomial conventions from the authoritative helper.
    # Small batches avoid constructing a q-by-q dummy protograph.
    mask = (1 << q)-1
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start+16, q)))
        monomials = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j*q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j*q)) & mask for row in zz])

    def verified_component(grid, side):
        # GF2 linear assembly of verified monomial blocks; the final
        # complete construction independently checks this assembly.
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    block = blocks[side][G.idx(g)]
                    for r, word in enumerate(block):
                        rows[i*q+r] ^= word << (j*q)
        columns = [0]*width
        for check, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << check
        return rows, columns

    local_cache = {}

    def localized_word(columns, key):
        """Exact localized minimum for nullity<=14; witnesses otherwise."""
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
            local = random.Random(
                diagnostic_seed+sum((i+1)*g
                                    for i, g in enumerate(key[1])))
            for _ in range(128):
                word = 0
                for other in local.sample(basis, min(4, len(basis))):
                    word ^= other
                note(word)
        if best is not None:
            value = 0
            for c in bits(best):
                value ^= columns[c]
            assert value == 0
        local_cache[key] = best
        return best

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
            return width+1, None
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        return best, witness

    def merit(record):
        return (tuple(-x for x in record["small"]),
                record["d"],
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        support = [[3, 3]] if side == "A" else support_b()
        profiles = plans[side]
        seed = diagnostic_seed+100003*side_id
        records, signatures, bank = [], set(), set()

        def survives(record, word):
            value = 0
            for c in bits(word):
                value ^= record["columns"][c]
            return value == 0

        def refresh(record):
            for word in bank:
                if word.bit_count() < record["d"] and survives(record, word):
                    record["d"], record["word"] = word.bit_count(), word

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
            words = []
            for j, cell in enumerate(grid[0]):
                key = (side, tuple(sorted(G.idx(g) for g in cell)))
                word = localized_word(columns[j*q:(j+1)*q], key)
                if word is not None:
                    words.append(word << (j*q))
            word = min(words, key=int.bit_count) if words else None
            reading = (word.bit_count() if word is not None
                       else len(columns)+1)
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "cycles": stats[1], "trial": trial,
                "rank": gf2_rank(rows), "localized": reading,
                "local_words": words, "d": reading, "word": word,
            }

        def diagnose(record, trials, offset):
            diagnostic = random.Random(seed+offset)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            if word is not None:
                bank.add(word)
                if reading < record["d"]:
                    record["d"], record["word"] = reading, word
            bank.update(record["local_words"])
            if len(bank) > 128:
                retained = sorted(
                    bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(retained)
            refresh(record)
            if record["word"] is not None:
                assert survives(record, record["word"])

        for trial in range(24):
            grid = [[fresh_cell(count, profiles[j])
                     for j, count in enumerate(row)]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], -r["localized"], r["cycles"], -r["rank"]))
        selected = records[:6]
        if not selected:
            raise ValueError("empty affine component shortlist")
        for record in selected:
            diagnose(record, 8, 0)
        selected.sort(key=score, reverse=True)

        # Two independent chains change actual labels in columns used
        # by verified words. No symmetry images are submitted as codes.
        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(8):
                refresh(current)
                active = (sorted({c//q for c in bits(current["word"])})
                          if current["word"] is not None else [])
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                choices = list(range(len(profiles)))
                changed = set()
                for move in range(2 if step % 4 == 3 else 1):
                    targeted = [j for j in active if j not in changed]
                    available = [j for j in choices if j not in changed]
                    if not available:
                        break
                    j = rng.choice(targeted if targeted and
                                   rng.random() < 0.85 else available)
                    changed.add(j)
                    grid[0][j] = fresh_cell(3, profiles[j])
                proposed = inspect(grid, 100+8*chain+step)
                if proposed is None or proposed["small"] > current["small"]:
                    continue
                if proposed["cycles"][0] > current["cycles"][0]+2*q:
                    continue
                diagnose(proposed, 6, 300007)
                selected.append(proposed)
                pk, ck = score(proposed), score(current)
                if pk >= ck or (
                        pk[:2] == ck[:2] and rng.random() < 0.12):
                    current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            # Readings accumulate independently for each actual matrix.
            diagnose(record, 48+8*(cohort % 3), 700001)
            diagnose(record, 16, 900001)
        finalists.sort(key=score, reverse=True)
        for record in finalists:
            refresh(record)
            assert record["word"] is not None
            assert survives(record, record["word"])
        return finalists

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]/(2*q)-ab[1]["cycles"][0]/(3*q)),
        reverse=True)

    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 7*q == 1400
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 2*q and 11*n <= 40*k < 13*n):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3, 3]] and sb == [[3, 3, 3]]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12
        # Confirm cached GF2 assemblies against the authoritative full lift.
        actual_a = [
            sum(((row >> (j*3*q)) & mask) << (j*q)
                for j in range(2))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (3*q))-1) for row in hz[:q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert css_ok(hx, hz)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "coset_profiles": plans,
            "component_readings": (a["d"], b["d"]),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
        }
    raise ValueError("affine pairings missed the exact 0.300 rate gate")


def NEW_CTX(rng, G, idx):
    """Twenty multirow support cards with verified kernel/logical repair."""
    from functools import lru_cache
    from itertools import product

    q = G.q
    assert q == 36
    assert A_SHAPE == (2, 5) and B_SHAPE == (3, 7)
    card, cohort = idx % 20, idx // 20
    concentration, density = divmod(card, 5)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # These small dummy constructions are never submitted. Both lift
    # conventions come directly from the authoritative scaffold helper.
    monomials = [[[g] for g in G.elements]]
    raw_a, _, _ = lifted_matrices(G, monomials, [[[]]])
    _, raw_b, _ = lifted_matrices(G, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def supports():
        # Exact A margins, including the prescribed concentrated columns.
        choices = [
            v for v in product(range(4), repeat=5)
            if sum(v) == 7
            and sum(x in (0, 3) for x in v) == concentration
        ]
        upper = rng.choice(choices)
        sa = [list(upper), [3-x for x in upper]]

        heavy = density+1
        degrees = [4]*heavy+[3]*(7-heavy)
        total = sum(degrees)
        margins = tuple(total//3+int(i < total % 3)
                        for i in range(3))
        # Concentration changes support topology, not a labeling gauge.
        sparse = set(range(min(7, 2*concentration)))
        options = []
        for j, degree in enumerate(degrees):
            candidates = [
                v for v in product(range(4), repeat=3)
                if sum(v) == degree
                and sum(x > 0 for x in v) == (2 if j in sparse else 3)
            ]
            rng.shuffle(candidates)
            options.append(candidates)

        @lru_cache(None)
        def feasible(j, remaining):
            if min(remaining) < 0:
                return False
            if j == 7:
                return remaining == (0, 0, 0)
            return any(feasible(
                j+1, tuple(remaining[i]-v[i] for i in range(3)))
                for v in options[j])

        if not feasible(0, margins):
            # Keep the card's degrees and A concentration, but relax B
            # concentration if its requested exact margins are infeasible.
            options = [
                [v for v in product(range(4), repeat=3)
                 if sum(v) == degree and sum(x > 0 for x in v) >= 2]
                for degree in degrees
            ]
            for candidates in options:
                rng.shuffle(candidates)
            feasible.cache_clear()
        assert feasible(0, margins)
        remaining = margins
        sb = [[], [], []]
        for j in range(7):
            candidates = [
                v for v in options[j]
                if feasible(j+1, tuple(
                    remaining[i]-v[i] for i in range(3)))
            ]
            v = rng.choice(candidates)
            for i in range(3):
                sb[i].append(v[i])
            remaining = tuple(remaining[i]-v[i] for i in range(3))
        assert [sum(row) for row in sa] == [7, 8]
        assert max(sum(row) for row in sb) <= 9
        assert _grid_weights(sa, sb) <= 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        return sa, sb

    sa, sb = supports()
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    orders = []
    for g in range(q):
        value = identity
        for order in range(1, q+1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group law")

    def legal(cell):
        ids = [G.idx(g) for g in cell]
        if len(set(ids)) != len(ids):
            return False
        if len(ids) <= 1:
            return True
        ratios = [
            table[inverses[a]][b]
            for a in ids for b in ids if a != b
        ]
        if 2 in (orders[g] for g in ratios):
            return False
        # Whole-cell Sidon geometry excludes internal repeated differences.
        if len(set(ratios)) != len(ratios):
            return False
        if len(ids) >= 3:
            roots = [table[inverses[ids[0]]][g] for g in ids[1:]]
            reached, todo = {identity}, [identity]
            for value in todo:
                for root in roots:
                    image = table[value][root]
                    if image not in reached:
                        reached.add(image)
                        todo.append(image)
            return len(reached) == q
        return True

    def draw(count, side, i, j):
        if not count:
            return []
        if count == 1:
            return [G.pick(rng)]
        # Binomial ratios alternate normal and primitive-complement types.
        # Cubics bridge adjacent complement fibers and generate all F9:C4.
        normal_pair = (i+2*j+density+cohort+int(side == "B")) % 3 == 0
        for _ in range(1024):
            phase = rng.randrange(4)
            if count == 2:
                ts = (phase, phase if normal_pair else (phase+1) % 4)
            else:
                ts = (phase, phase, (phase+1) % 4)
            cell = [(rng.randrange(3), rng.randrange(3), t) for t in ts]
            if legal(cell):
                return cell
        raise ValueError("multirow cell geometry exhausted")

    def fresh(side):
        support = sa if side == "A" else sb
        return [[draw(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for r, word in enumerate(blocks[side][G.idx(g)]):
                        rows[i*q+r] ^= word << (j*q)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None

        short, pairs = set(), {}
        singles = {value: 1 << c for c, value in enumerate(columns)}
        for c, value in enumerate(columns):
            for d in range(c):
                combined = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(combined)
                if old is not None:
                    short.add(word ^ old)
                old = pairs.get(combined)
                if old is not None:
                    short.add(word ^ old)
                else:
                    pairs[combined] = word

        uses, four = {}, 0
        for column in columns:
            checks = list(bits(column))
            for at, a in enumerate(checks):
                for b in checks[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        rank = gf2_rank(rows)
        if rank < len(rows)-6:
            return None
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, four=four, small=len(short),
            d=min((w.bit_count() for w in short), default=width+1),
            words=sorted(short, key=lambda w: (w.bit_count(), w))[:20],
        )

    def note(rec, word):
        if not word:
            return
        rec["d"] = min(rec["d"], word.bit_count())
        if word.bit_count() <= rec["d"]+6:
            rec["words"].append(word)
            if len(rec["words"]) > 64:
                rec["words"] = sorted(
                    set(rec["words"]),
                    key=lambda w: (w.bit_count(), w))[:20]

    def query(rec, trials, diagnostic_seed):
        """Actual component null words; every reading is an upper bound."""
        diagnostic = random.Random(diagnostic_seed)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = rec["words"][:]
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
                    note(rec, word)
            light = sorted(set(basis+retained),
                           key=lambda w: (w.bit_count(), w))[:20]
            for word in light:
                for other in basis+retained:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= rec["d"]+2:
                        note(rec, candidate)
            moves = basis+retained
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(rec, word)
            rec["words"] = sorted(
                set(rec["words"]),
                key=lambda w: (w.bit_count(), w))[:20]
            retained = rec["words"]
            if rec["d"] <= 4:
                break
        assert rec["words"]
        assert all(syndrome(columns, w) == 0 for w in rec["words"])

    def merit(rec):
        return (-rec["small"], rec["d"],
                -sum(w.bit_count() == rec["d"] for w in rec["words"]),
                -rec["four"], rec["rank"])

    def signature(rec):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in rec["grid"] for cell in row)

    def pool(side):
        offset = 100003*int(side == "B")
        records, seen = [], set()
        for _ in range(96):
            rec = inspect(fresh(side), side)
            if rec is None or signature(rec) in seen:
                continue
            seen.add(signature(rec))
            records.append(rec)
        records.sort(key=lambda r: (
            r["small"], r["four"], -r["rank"]))
        selected = records[:8]
        if not selected:
            raise ValueError("empty multirow component pool")
        for rec in selected:
            query(rec, 8, seed+offset)
        selected.sort(key=merit, reverse=True)

        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(24):
                grid = copy_grid(current["grid"])
                support = sa if side == "A" else sb
                occupied = ({c//q for c in bits(current["words"][0])}
                            if current["words"] else set())
                positions = [
                    (i, j) for i, row in enumerate(support)
                    for j, count in enumerate(row)
                    if count and (j in occupied or step % 4 == 3)
                ]
                if not positions:
                    positions = [
                        (i, j) for i, row in enumerate(support)
                        for j, count in enumerate(row) if count]
                i, j = rng.choice(positions)
                old = grid[i][j]
                if len(old) >= 2 and step % 4:
                    proposed = list(old)
                    t = rng.randrange(len(old))
                    g = old[t]
                    proposed[t] = (
                        rng.randrange(3), rng.randrange(3), g[2])
                    if proposed == old or not legal(proposed):
                        continue
                    grid[i][j] = proposed
                else:
                    grid[i][j] = draw(len(old), side, i, j)
                rec = inspect(grid, side)
                if rec is None or signature(rec) in seen:
                    continue
                seen.add(signature(rec))
                if rec["small"] > current["small"]:
                    continue
                for word in current["words"]:
                    if syndrome(rec["columns"], word) == 0:
                        note(rec, word)
                query(rec, 6, seed+offset+300007+7919*step)
                selected.append(rec)
                if merit(rec) >= merit(current) or (
                        rec["d"] == current["d"] and rng.random() < 0.08):
                    current = rec

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for rec in finalists:
            query(rec, 48+8*(cohort % 3), seed+offset+700001)
            query(rec, 24, seed+offset+1700003)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, pivots):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def localized(hx, hz, a, b, n):
        """Syndrome-verified logicals reduced with ORIGINAL stabilizers."""
        readings = []
        for kind, opposite, stabilizers, rec in (
                ("Z", hx, hz, a), ("X", hz, hx, b)):
            pivots = span(stabilizers)
            candidates = []
            for word in rec["words"][:8]:
                for block in range(7 if kind == "Z" else 5):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((7*j+block)*q)
                            for j in range(5))
                    else:
                        actual = word << (7*block*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if nontrivial(actual, pivots):
                        candidates.append(actual)
            reading = n+1
            diagnostic = random.Random(
                seed+3100009+100003*int(kind == "X"))
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers,
                    key=lambda row: (actual & row).bit_count(),
                    reverse=True)[:32]
                for _ in range(128):
                    u, v = diagnostic.sample(neighborhood, 2)
                    candidate = actual ^ u ^ v
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and nontrivial(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    best = None
    for restart in range(2):
        aa, bb = pool("A"), pool("B")
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                assert n == 1476 and k >= 432
                if not (275*n <= 1000*k < 325*n):
                    continue
                actual_a = [
                    sum(((hx[(i*7)*q+t] >> (j*7*q)) & mask) << (j*q)
                        for j in range(5))
                    for i in range(2) for t in range(q)]
                actual_b = [
                    row & ((1 << (7*q))-1) for row in hz[:3*q]]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = localized(hx, hz, a, b, n)
                detected = min(a["d"], b["d"], *logicals)
                key = (
                    -a["small"]-b["small"], k*detected*detected,
                    detected, a["d"]+b["d"], -a["four"]-b["four"])
                if best is None or key > best[0]:
                    best = (key, A, B, hx, hz, k,
                            (a["d"], b["d"]), logicals)
        if best is not None:
            break
    if best is None:
        raise ValueError("multirow pairings missed exact cell 0.300 gate")
    key, A, B, hx, hz, k, components, logicals = best
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert css_ok(hx, hz)
    return dict(
        idx=idx, cells={"A": A, "B": B}, k=k,
        card=(concentration, density, cohort),
        supports=(sa, sb), component_readings=components,
        verified_localized_logical_readings=logicals, quality=key)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected multirow cell, including zero blocks."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) <= 5
    assert len({G._key(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Full relative generation with localized-kernel and surviving-witness repair removes the order-200 single-row weight-sixteen bottleneck; all five measured parent codes instead retained sector-1 X minima of sixteen.
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
