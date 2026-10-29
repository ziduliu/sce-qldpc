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
CONCEPT = ("Hypothesis: on split-metacyclic C67:C3(s=29), order 201, A=(2,1), B=(3,1), whole-cell Sidon supports, five degree allocations and independently challenged transpose kernels improve cell 0.300 beyond recorded fitness 73.143. Search readings remain provisional upper bounds, not distance certificates.")
GROUP = ("Keep the parent's split-metacyclic construction formula and GTYPE, resizing only m,r,s to 67,3,29 under the L2 size allowance. Here 29^2=37 and 29^3=1 modulo 67, with faithful order-three action. Factor seven permits q<=214; 201 is the largest prime-normal faithful C3 lift in that window. The related order-201 wide bed reached 59.4 and the order-291 tall bed reached 965.9 at another rate; neither establishes distance transfer.")
SHAPES = ("Change A to (2,1), B to (3,1): n=7*201=1407, k>=402, guaranteed rate 2/7=0.285714 target cell 0.300. Raw components are tall; distance-bearing transposes have variables/checks 402/201 and 603/201, nominal rates 1/2 and 2/3. Parent measured minima were sector-2 X witnesses at weights 18..21 across Arow/Brow blocks, while Z readings were 30..47. Replace both kernels and enforce the exact full-rank interval [0.275,0.325).")
FILLING = ("Twenty cards cross five degree allocations with four genuine coset-occupancy prescriptions. A/B count pairs are (4,4)/(2,3,3), (4,4)/(2,2,4), (3,3)/(3,3,3), (3,3)/(2,3,4), and (3,4)/(2,3,3); both CSS maxima are at most twelve. Binomial ratios have order 67. Larger cells require distinct directed differences in BOTH orientations and noncommuting relative generators. Verified transpose lifts screen connectivity, ranks and all kernels through four. Restricted enumeration, accumulated circuit cancellation, witness-directed exponent repair and independent finalist challenges select new labels; extra repair targets the higher-rate B transpose. Full native-lift agreement, original-span-verified localized logicals, ranks, weights and CSS gate pairing. No heuristic certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_m67r3s29_21x31_whole_sidon_degreecards_cokernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 201

def custom_ops(q):
    # Same split-metacyclic construction rule; L2 parameter resize only.
    # 29^2=37 and 29^3=1 modulo 67.
    m, r, s = 67, 3, 29
    if not valid_q(q) or pow(s, r, m) != 1:
        raise ValueError("invalid order-201 metacyclic parameters")
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (3, 1)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty degree/coset cards with challenged transpose-component kernels."""
    q = G.q
    assert q == 201
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert (ra, ca, rb, cb) == (2, 1, 3, 1)
    m = q // 3
    card, cohort = idx % 20, idx // 20
    allocation, model = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    # Distinct weighted support graphs, not row-permutation copies.
    degree_plans = (
        ((4, 4), (2, 3, 3)),
        ((4, 4), (2, 2, 4)),
        ((3, 3), (3, 3, 3)),
        ((3, 3), (2, 3, 4)),
        ((3, 4), (2, 3, 3)),
    )
    acounts, bcounts = degree_plans[allocation]
    sa = [[count] for count in acounts]
    sb = [[count] for count in bcounts]
    wx = max(acounts) + sum(bcounts)
    wz = max(bcounts) + sum(acounts)
    assert max(wx, wz) == _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(2 <= count <= 4 for count in acounts + bcounts)

    # Cache native operations only. All monomial matrices below come
    # from lifted_matrices, not an independently implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = G.idx(G.e)

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    blocks = {"A": [], "B": []}
    for start in range(0, q, 12):
        ids = list(range(start, min(start + 12, q)))
        monomials = [[[g] for g in ids]]
        _, zz, _ = lifted_matrices(probe, monomials, [[[]]])
        xx, _, _ = lifted_matrices(probe, [[[]]], monomials)
        shift = len(ids) * q
        for j in range(len(ids)):
            blocks["A"].append([
                zz[j*q+r] >> shift for r in range(q)])
            blocks["B"].append([
                xx[j*q+r] >> shift for r in range(q)])

    def whole_sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        ids = [G.idx(g) for g in cell]
        left, right = set(), set()
        for a in ids:
            for b in ids:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(ids) * (len(ids) - 1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if len(ids) >= 3:
            ratios = [table[inverses[ids[0]]][g] for g in ids[1:]]
            if not any(table[a][b] != table[b][a]
                       for t, a in enumerate(ratios)
                       for b in ratios[:t]):
                return False
        return True

    def draw_cell(count, side, i, j):
        # Card prescriptions change within-cell multiplicities, not
        # merely the names of quotient fibers or a common group gauge.
        if count == 2:
            pattern = (0, 0)       # Every binomial ratio has order 67.
        elif count == 3:
            if model < 2:
                pattern = (0, 0, 1)
            elif model == 2:
                pattern = (0, 1, 2)
            else:
                pattern = ((0, 1, 2) if
                           (i + int(side == "B") + cohort) % 2 == 0
                           else (0, 0, 1))
        elif count == 4:
            recipes = ((0, 0, 1, 2), (0, 0, 0, 1),
                       (0, 0, 1, 1))
            kind = model if model < 3 else (
                1 if (i + int(side == "B") + cohort) % 2 == 0 else 2)
            pattern = recipes[kind]
        else:
            raise ValueError("unplanned cell degree")

        for _ in range(1024):
            phase = rng.randrange(3)
            bs = [(phase + b) % 3 for b in pattern]
            cell = []
            for b in sorted(set(bs)):
                aa = rng.sample(range(m), bs.count(b))
                cell.extend((a, b) for a in aa)
            if whole_sidon(cell):
                assert len(cell) == count
                return cell
        raise ValueError("whole-cell Sidon prescription exhausted")

    def fresh(side):
        support = sa if side == "A" else sb
        return [[draw_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def component(grid, side):
        # Assemble the transpose component by GF2 linearity from
        # authoritative monomial blocks. Final native lifts check it.
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nc * q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    block = blocks[side][G.idx(g)]
                    for r, word in enumerate(block):
                        rows[j*q+r] ^= word << (i*q)
        columns = [0] * (nr * q)
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return rows, columns

    def inspect(grid, side):
        rows, columns = component(grid, side)
        if not all(columns) or len(set(columns)) != len(columns):
            return None
        rank = gf2_rank(rows)
        if rank < len(rows) - 6:
            return None
        adjacency = [0] * len(rows)
        uses, four = {}, 0
        for column in columns:
            neighbors = list(bits(column))
            for at, a in enumerate(neighbors):
                adjacency[a] |= column ^ (1 << a)
                for b in neighbors[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows)) - 1:
            return None
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cycles": four,
            "signature": tuple(tuple(sorted(G.idx(g) for g in cell))
                               for row in grid for cell in row),
            "d": len(columns) + 1, "words": [],
        }

    def kernel4_free(columns):
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        pairs = set()
        for i, a in enumerate(columns):
            for b in columns[:i]:
                syndrome = a ^ b
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    def basis_of(columns, order):
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

    def localized(record):
        # Exact only within each inspected one-/two-block restriction
        # when its nullity is <=12. Larger spaces supply witnesses only.
        columns = record["columns"]
        count = len(columns) // q
        restrictions = [(i,) for i in range(count)]
        restrictions += [(i, j) for i in range(count)
                         for j in range(i + 1, count)]
        words = []
        for active in restrictions:
            positions = [c for i in active for c in range(i*q, (i+1)*q)]
            basis = basis_of(columns, positions)
            if not basis:
                continue
            if len(basis) <= 12:
                word = 0
                best = None
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    if best is None or word.bit_count() < best.bit_count():
                        best = word
                words.append(best)
            else:
                words.extend(sorted(basis, key=int.bit_count)[:8])
        record["words"] = sorted(set(words), key=int.bit_count)[:20]
        if record["words"]:
            record["d"] = min(record["d"],
                              record["words"][0].bit_count())

    def query(record, side, trials, offset):
        diagnostic = random.Random(seed + offset +
                                   (100003 if side == "A" else 200003))
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = list(record["words"])
        reading = record["d"]
        for trial in range(trials):
            diagnostic.shuffle(order)
            basis = basis_of(columns, order)
            assert basis
            light = sorted(set(retained + basis), key=int.bit_count)[:24]
            reading = min(reading, light[0].bit_count())
            additions = light[:]
            for i, word in enumerate(basis):
                # Complete circuit-pair cancellation regularly challenges
                # B; intervening trials also cancel accumulated light words.
                others = (basis[:i] if side == "A" or trial % 4 == 0
                          else light + retained)
                for other in others:
                    candidate = word ^ other
                    if not candidate:
                        continue
                    weight = candidate.bit_count()
                    reading = min(reading, weight)
                    if weight <= reading + 4:
                        additions.append(candidate)
                        if len(additions) > 128:
                            additions = sorted(set(additions),
                                               key=int.bit_count)[:24]
            moves = retained + basis
            diagnostic.shuffle(moves)
            for initial in light[:4]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                additions.append(word)
                reading = min(reading, word.bit_count())
            retained = sorted(set(retained + additions),
                              key=int.bit_count)[:24]
            if reading <= 8:
                break
        record["d"] = min(reading, retained[0].bit_count())
        record["words"] = retained
        assert all((retained[0] & row).bit_count() % 2 == 0
                   for row in record["rows"])

    def quality(record):
        return (record["d"], -record["cycles"], record["rank"])

    def select(side):
        records, signatures = [], set()
        screened = []
        for batch in range(3):
            for _ in range(64):
                record = inspect(fresh(side), side)
                if record is None or record["signature"] in signatures:
                    continue
                signatures.add(record["signature"])
                records.append(record)
            records.sort(key=lambda r: (r["cycles"], -r["rank"]))
            for record in records:
                if not kernel4_free(record["columns"]):
                    continue
                localized(record)
                query(record, side, 6, 0)
                screened.append(record)
                if len(screened) >= 8:
                    break
            if screened:
                break
        if not screened:
            raise ValueError("balanced transpose component screening exhausted")

        finalists = sorted(screened, key=quality, reverse=True)[:3]
        for record in finalists:
            query(record, side, 20, 700001)
        finalists.sort(key=quality, reverse=True)
        current = finalists[0]

        for move in range(12 if side == "A" else 16):
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            active = sorted({c // q for c in bits(current["words"][0])})
            i = rng.choice(active)
            j = rng.randrange(len(grid[0]))
            if move % 5 == 0:
                grid[i][j] = draw_cell(len(grid[i][j]), side, i, j)
            else:
                t = rng.randrange(len(grid[i][j]))
                old = grid[i][j][t]
                available = [(a, old[1]) for a in range(m)
                             if (a, old[1]) not in grid[i][j]]
                grid[i][j][t] = rng.choice(available)
                if not whole_sidon(grid[i][j]):
                    continue
            proposed = inspect(grid, side)
            if proposed is None or proposed["signature"] in signatures:
                continue
            signatures.add(proposed["signature"])
            if not kernel4_free(proposed["columns"]):
                continue
            localized(proposed)
            query(proposed, side, 6, 0)
            if quality(proposed) <= quality(current):
                continue
            query(proposed, side, 20, 700001)
            if quality(proposed) > quality(current):
                current = proposed

        if all(current["signature"] != r["signature"] for r in finalists):
            finalists.append(current)
        finalists = sorted(finalists, key=quality, reverse=True)[:3]
        for record in finalists:
            query(record, side, 64, 1700009)
            query(record, side, 24, 2700011)
        return sorted(finalists, key=quality, reverse=True)[:2]

    aa, bb = select("A"), select("B")

    def span_of(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, span):
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_reading(hx, hz, a, b, n):
        n1 = ca * cb * q
        readings = []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            span = span_of(stabilizers)
            retained = []
            for word in record["words"][:8]:
                for anchor in range(rb if kind == "X" else ra):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << (n1 + (i*rb+anchor)*q)
                            for i in range(ra))
                    else:
                        actual = word << (n1 + anchor*rb*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if nontrivial(actual, span):
                        retained.append(actual)
            retained.sort(key=int.bit_count)
            found = n + 1
            diagnostic = random.Random(seed +
                                       (300007 if kind == "X" else 400009))
            for actual in retained[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                neighbors = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(64):
                    first, second = diagnostic.sample(neighbors, 2)
                    candidate = actual ^ first ^ second
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and nontrivial(actual, span)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                found = min(found, actual.bit_count())
            readings.append(found)
        return tuple(readings)

    best = None
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1407 and k >= q*(ca-ra)*(cb-rb)
                    and 275*n <= 1000*k < 325*n):
                continue
            readings = logical_reading(hx, hz, a, b, n)
            detected = min(a["d"], b["d"], *readings)
            merit = (detected, k*detected*detected,
                     a["d"] + b["d"], -a["cycles"] - b["cycles"])
            if best is None or merit > best[0]:
                best = (merit, a, b, hx, hz, k, readings)
    if best is None:
        raise ValueError("Sidon degree-card pairings missed cell 0.300")

    merit, a, b, hx, hz, k, readings = best
    n1 = ca * cb * q
    actual_a = [
        sum(((hz[j*rb*q+r] >> (n1+i*rb*q)) & mask) << (i*q)
            for i in range(ra))
        for j in range(ca) for r in range(q)]
    actual_b = [row >> n1 for row in hx[:cb*q]]
    assert actual_a == a["rows"] and actual_b == b["rows"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert [[len(cell) for cell in row] for row in a["grid"]] == sa
    assert [[len(cell) for cell in row] for row in b["grid"]] == sb
    assert all(whole_sidon(cell)
               for grid in (a["grid"], b["grid"])
               for row in grid for cell in row)
    return {
        "idx": idx, "cells": {"A": a["grid"], "B": b["grid"]},
        "card": (allocation, model, cohort), "supports": (sa, sb), "k": k,
        "screen": merit, "component_readings": (a["d"], b["d"]),
        "verified_localized_logical_readings": readings,
        "component_ranks": (a["rank"], b["rank"]),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without post-screen randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation with rolled positive-cell supports excludes abundant light sector-2 logicals on Z19:Z3 (4,2)x(5,3); measured slots 2 and 9 instead contain X logicals of weight six.
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
