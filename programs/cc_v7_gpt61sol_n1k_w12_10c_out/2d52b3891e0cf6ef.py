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
CONCEPT = ("On Z19:Z3(s=7), order 57, A=(4,2), B=(5,3), balanced support cards and long-relative-order binomials with independently retested sector-2 kernels should improve five-code fitness beyond this bed's recorded 79.3. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Frozen order-57 metacyclic group: this exact bed has reached fitness 79.3, versus the current 15.7088. Same-coset binomial ratios have order 19 rather than three. Retain the validated group while testing whether that relative geometry and stronger label selection suppress the observed logical families.")
SHAPES = ("Frozen A=(4,2), B=(5,3): n=1482, k>=228 and guaranteed rate 2/13 target cell 0.150; q_max=57. Original components have variables/checks 114/228 and 171/285; their transposes have 228/114 and 285/171. Parent minima were sector-2 X witnesses, including weight-six families over Arow/Brow blocks and weight 11..13 families spread across most blocks. Exact full ranks retain [0.125,0.175).")
FILLING = ("Twenty canonical support pairs have A-column totals seven, A-row totals three/four, B-row totals four/five and B-column totals at most eight, giving both CSS row caps twelve. Fresh labels prefer order-19 binomial ratios and contrast actual trinomial coset occupancies. Authoritative transposed component lifts screen connectivity, ranks and kernels through four; localized enumeration, repeated kernel searches and witness-directed label repair challenge the sector-2 bottleneck. Independent finalist retests and original-span-verified localized quantum witnesses select pairings. All sampled readings remain upper bounds.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m19r3s7_42x53_balanced_longratio_cokernel_retest_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    # metacyclic Z_19 x| Z_3: y x y^-1 = x^7 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 19, 3, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (4, 2)               # (ra, ca)
B_SHAPE = (5, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Balanced support cards and independently challenged tall components."""
    from itertools import product, permutations, combinations_with_replacement

    q = G.q
    assert q == 57
    card, cohort = idx % 20, idx // 20
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def canonical(rows):
        # Original-row permutations and original-column permutations.
        return min(
            tuple(sorted(tuple(row[j] for j in order) for row in rows))
            for order in permutations(range(len(rows[0]))))

    ap = set()
    arows = [row for row in product(range(1, 4), repeat=2)
             if sum(row) in (3, 4)]
    for rows in combinations_with_replacement(arows, 4):
        if sum(map(sum, rows)) != 14:
            continue
        if all(sum(row[j] for row in rows) == 7 for j in range(2)):
            ap.add(canonical(rows))

    bp = set()
    brows = [row for row in product(range(1, 4), repeat=3)
             if sum(row) in (4, 5)]
    for rows in combinations_with_replacement(brows, 5):
        total = sum(map(sum, rows))
        columns = [sum(row[j] for row in rows) for j in range(3)]
        if 22 <= total <= 24 and max(columns) <= 8:
            bp.add(canonical(rows))

    plans = [(a, b) for a in sorted(ap) for b in sorted(bp)]
    plans.sort(key=lambda ab: (
        -sum(map(sum, ab[1])),
        sum(value == 1 for grid in ab for row in grid for value in row),
        sum(value * value for grid in ab for row in grid for value in row),
        ab))
    assert len(plans) >= 20
    sa, sb = ([list(row) for row in grid] for grid in plans[card])
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(sum(sa[i][j] for i in range(4)) == 7 for j in range(2))
    assert all(sum(row) in (3, 4) for row in sa)
    assert all(sum(row) in (4, 5) for row in sb)

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

    def draw_cell(count, side, i, j):
        if count == 1:
            return [G.pick(rng)]
        phase = rng.randrange(3)
        if count == 2:
            # Most ratios lie in C19. Some B cards deliberately test
            # mixed-coset ratios instead; these are not global gauges.
            mixed = (side == "B" and (card + cohort) % 4 == 3
                     and (i + 2*j) % 3 == 0)
            bs = [phase, (phase + 1) % 3] if mixed else [phase, phase]
        else:
            model = (card + 2*i + j + cohort) % 3
            patterns = ((0, 0, 0), (0, 0, 1), (0, 1, 2))
            bs = [(phase + b) % 3 for b in patterns[model]]
        cell = []
        for b in sorted(set(bs)):
            aa = rng.sample(range(19), bs.count(b))
            cell.extend((a, b) for a in aa)
        assert len(cell) == len(set(cell)) == count
        return cell

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
        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = basis_of(columns, order)
            assert basis
            light = sorted(set(retained + basis), key=int.bit_count)[:16]
            reading = min(reading, light[0].bit_count())
            additions = light[:]
            for i, word in enumerate(basis):
                for other in basis[:i]:
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
            for _ in range(40):
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

        for move in range(10 if side == "A" else 6):
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            active = sorted({c // q for c in bits(current["words"][0])})
            i = rng.choice(active)
            j = rng.randrange(len(grid[0]))
            t = rng.randrange(len(grid[i][j]))
            old = grid[i][j][t]
            available = [(a, old[1]) for a in range(19)
                         if (a, old[1]) not in grid[i][j]]
            grid[i][j][t] = rng.choice(available)
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
            query(record, side, 48, 1700009)
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
        n1 = 6 * q
        readings = []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            span = span_of(stabilizers)
            retained = []
            for word in record["words"][:5]:
                for anchor in range(5 if kind == "X" else 4):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << (n1 + (i*5+anchor)*q)
                            for i in range(4))
                    else:
                        actual = word << (n1 + anchor*5*q)
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
            if not (n == 1482 and k >= 228
                    and 125*n <= 1000*k < 175*n):
                continue
            readings = logical_reading(hx, hz, a, b, n)
            detected = min(a["d"], b["d"], *readings)
            merit = (detected, k*detected*detected,
                     a["d"] + b["d"], -a["cycles"] - b["cycles"])
            if best is None or merit > best[0]:
                best = (merit, a, b, hx, hz, k, readings)
    if best is None:
        raise ValueError("balanced pairings missed cell 0.150")

    merit, a, b, hx, hz, k, readings = best
    n1 = 6 * q
    actual_a = [
        sum(((hz[j*5*q+r] >> (n1+i*5*q)) & mask) << (i*q)
            for i in range(4))
        for j in range(2) for r in range(q)]
    actual_b = [row >> n1 for row in hx[:3*q]]
    assert actual_a == a["rows"] and actual_b == b["rows"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    return {
        "idx": idx, "cells": {"A": a["grid"], "B": b["grid"]},
        "card": (card, cohort), "supports": (sa, sb), "k": k,
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
