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
CONCEPT = ("For metacyclic Z37:Z3(s=10), order 111, A=(1,2), B=(3,5), degree-nine A components paired with degree-three B columns, alongside independently optimized degree-eight controls, should produce five measured distance readings above 24 and improve cell 0.150 beyond fitness 79.4716. Readings remain provisional upper bounds.")
GROUP = ("Frozen Z37:Z3(s=10), q=111: this bed scores 79.4716. Its five measured X readings were 74..85, whereas its minimum Z readings were 22..24. That measured imbalance motivates reallocating filling budgets and strengthening label selection, not changing the group or claiming certified distance.")
SHAPES = ("Frozen A=(1,2), B=(3,5): n=13*111=1443, k>=222 and guaranteed rate 2/13; the size window permits q<=115. Unequal components have 222 variables/111 checks and 555 variables/333 checks. Parent minimum Z witnesses used both Acols at one Bcol, especially Bcol1..3. Protect the A kernel while screening B against loss of its large measured margin; exact full ranks retain [0.125,0.175).")
FILLING = ("Twenty cards cross four A degree/coset models with five B topologies. A=(4,5) uses B-column degree three; A=(4,4) and (3,5) controls use degree four. Both quantum row caps remain <=12. Independently shortlist verified component lifts by cycle geometry and ranks, exclude component kernels through four, search genuine kernel words repeatedly, repair A labels, and independently confirm finalists. Full ranks and actual-span-checked component-derived quantum witnesses select the final pair. Sampled component and quantum witnesses are upper readings, not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m37r3s10_12x35_degree9_kernel_descent_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same metacyclic rule: 10^3 = 1 mod 37, with action of order 3.
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


# Four component-degree budgets, each with five deliberate B topologies.
# Lists are read-only templates; every candidate gets fresh cell lists.
_A_DEGREES = ((3, 3), (3, 4), (4, 4), (3, 5))
_B_33 = (
    ((1,2,2,2,2), (2,1,2,2,2), (2,2,1,2,2)),
    ((3,1,1,2,2), (1,2,2,2,2), (2,2,2,1,2)),
    ((3,1,1,2,2), (1,3,2,1,2), (1,1,3,2,2)),
    ((3,0,2,2,2), (1,3,2,1,2), (2,2,1,3,1)),
    ((3,3,1,1,1), (1,1,3,3,1), (2,2,2,1,2)),
)
_B_34 = (
    ((1,1,2,2,2), (2,2,1,1,2), (2,2,2,1,1)),
    ((3,1,1,1,2), (1,2,2,2,1), (1,2,2,1,2)),
    ((3,1,1,1,2), (1,3,1,2,1), (1,1,2,2,2)),
    ((3,1,1,1,2), (1,3,1,2,1), (1,1,3,1,2)),
    ((3,0,2,1,2), (1,3,1,2,1), (1,2,2,1,2)),
)
_B_COL4 = (
    ((2,2,1,1,1), (1,1,2,2,1), (1,1,1,1,2)),
    ((3,1,1,1,1), (0,2,2,2,1), (1,1,1,1,2)),
    ((3,1,1,1,1), (0,2,3,2,0), (1,1,0,1,3)),
    ((2,2,2,1,0), (2,0,1,2,2), (0,2,1,1,2)),
    ((3,2,0,1,1), (1,0,2,3,1), (0,2,2,0,2)),
)
_B_DESIGNS = (_B_33, _B_34, _B_COL4, _B_COL4)


def _mixed_cell(rng, G, count, phase):
    """Distinct labels; multi-term cells span at least two quotient cosets."""
    if count == 0:
        return []
    patterns = {
        1: (0,),
        2: (0, 1),
        3: (0, 0, 1),
        4: (0, 0, 1, 2),
        5: (0, 0, 1, 1, 2),
    }
    r = max(g[1] for g in G.elements) + 1
    shift = (phase + rng.randrange(r)) % r
    bs = [(b + shift) % r for b in patterns[count]]
    cell = []
    for b in sorted(set(bs)):
        pool = [g for g in G.elements if g[1] == b]
        cell.extend(rng.sample(pool, bs.count(b)))
    return cell


def _component_stats(rows):
    """Count small classical kernels and four-cycles, not CSS distance."""
    columns = {}
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            pos = bit.bit_length() - 1
            columns[pos] = columns.get(pos, 0) | (1 << check)
            row ^= bit
    syndromes = list(columns.values())
    singles = {}
    n2 = 0
    four_cycles = 0
    check_pairs = {}
    for syndrome in syndromes:
        n2 += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
        neighbors = []
        x = syndrome
        while x:
            bit = x & -x
            neighbors.append(bit.bit_length() - 1)
            x ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                pair = (neighbors[b], neighbors[a])
                previous = check_pairs.get(pair, 0)
                four_cycles += previous
                check_pairs[pair] = previous + 1

    # With distinct nonzero columns, pair collisions count weight-4
    # kernels three times; membership in singles counts weight-3
    # kernels three times. Duplicate columns are separately penalized.
    pairs = {}
    n3 = 0
    n4 = 0
    for a, x in enumerate(syndromes):
        for b in range(a):
            syndrome = x ^ syndromes[b]
            n3 += singles.get(syndrome, 0)
            previous = pairs.get(syndrome, 0)
            n4 += previous
            pairs[syndrome] = previous + 1
    bad = 10000 * n2 + 100 * (n3 // 3) + n4 // 3
    return bad, four_cycles, len(syndromes)


def NEW_CTX(rng, G, idx):
    """Reallocate degree budgets and independently optimize verified kernels."""
    card, cohort = idx % 20, idx // 20
    model, topology = card % 4, card // 4
    q = G.q
    assert q == 111
    mask = (1 << q) - 1
    seed = rng.randrange(1 << 62)

    # Each column below totals three. All rows total at most six.
    # These are density/topology changes, not permutations of one labeling.
    b_col3 = (
        ((1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1)),
        ((2,0,1,1,1), (1,2,1,1,1), (0,1,1,1,1)),
        ((2,0,1,1,1), (1,2,0,1,1), (0,1,2,1,1)),
        ((2,2,0,1,1), (1,0,2,0,1), (0,1,1,2,1)),
        ((2,2,0,1,0), (1,0,2,0,2), (0,1,1,2,1)),
    )
    a_counts = ((4,5), (4,5), (4,4), (3,5))
    a_occupancies = (
        ((2,1,1), (2,2,1)),
        ((2,1,1), (3,1,1)),
        ((2,1,1), (3,1,0)),
        ((2,1,0), (2,2,1)),
    )
    sa = [list(a_counts[model])]
    source = b_col3 if model < 2 else _B_COL4
    sb = [list(row) for row in source[topology]]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= c <= 5 for grid in (sa, sb)
               for row in grid for c in row)

    # Cache native operations only. Every matrix is still constructed
    # by lifted_matrices, with its verified left/right conventions.
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

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def draw_cell(occupancy):
        fibers = list(range(3))
        rng.shuffle(fibers)
        cell = []
        for b, count in zip(fibers, occupancy):
            if not count:
                continue
            if count == 3 and cohort % 3:
                # Different three-point affine classes, not a global
                # automorphism applied to a previously submitted code.
                t = (3, 5)[cohort % 3 - 1]
                origin, scale = rng.randrange(37), rng.randrange(1, 37)
                aa = [origin, (origin + scale) % 37,
                      (origin + t * scale) % 37]
            else:
                for _ in range(128):
                    aa = rng.sample(range(37), count)
                    ds = [(a-c) % 37 for a in aa for c in aa if a != c]
                    if len(ds) == len(set(ds)):
                        break
                else:
                    raise ValueError("Sidon fiber generation failed")
            cell.extend((a, b) for a in aa)
        assert len(cell) == len(set(cell)) == sum(occupancy)
        return cell

    def fresh(side):
        if side == "A":
            return [[draw_cell(o) for o in a_occupancies[model]]]
        patterns = {
            0: (0,0,0), 1: (1,0,0), 2: (1,1,0),
            3: (1,1,1) if cohort % 3 == 2 else (2,1,0),
        }
        return [[draw_cell(patterns[count]) for count in row]
                for row in sb]

    def component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        width = len(grid[0]) * q
        # Empty dummy sides isolate actual components; these probes
        # are never submitted as codes.
        if side == "A":
            rows, _, _ = lifted_matrices(probe, indexed, [[[]]])
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], indexed)
        assert all(row >> width == 0 for row in rows)
        return rows, width

    def inspect(grid, side):
        rows, width = component(grid, side)
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None

        pair_uses = {}
        four = forced = 0
        adjacency = [0] * len(rows)
        for syndrome in columns:
            checks = list(bits(syndrome))
            degree = len(checks)
            forced += degree * (degree-1) * (degree-2) // 6
            for i, a in enumerate(checks):
                adjacency[a] |= syndrome ^ (1 << a)
                for b in checks[:i]:
                    key = (b, a)
                    previous = pair_uses.get(key, 0)
                    four += previous
                    pair_uses[key] = previous + 1

        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows)) - 1:
            return None

        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b] &
                              (-1 << (b+1))).bit_count()
        # Triangle excess is a heuristic when four-cycles are present.
        cycles = (four, max(0, triangles-forced))
        rank = gf2_rank(rows)
        floor = q-3 if side == "A" else 3*q-9
        if rank < floor:
            return None
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        return dict(grid=grid, rows=rows, columns=columns,
                    cycles=cycles, rank=rank, signature=signature)

    def kernel4_free(columns):
        """Exact exclusion of every nonzero classical kernel word <=4."""
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

    def kernel_query(columns, diagnostic_seed, trials):
        """Genuine kernel witnesses in original coordinates; upper readings."""
        diagnostic = random.Random(diagnostic_seed)
        width = len(columns)
        reading = width + 1
        retained = []
        order = list(range(width))
        for _ in range(trials):
            diagnostic.shuffle(order)
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
                    reading = min(reading, word.bit_count())
            if not basis:
                raise ValueError("component unexpectedly has no kernel")

            light = sorted(set(retained + basis),
                           key=int.bit_count)[:20]
            candidates = light[:]
            # Exhaustive basis pairs for the bottleneck A component;
            # light-by-basis pairs suffice for the larger B guard.
            for i, word in enumerate(basis):
                others = basis[:i] if width == 2*q else light[:16]
                for other in others:
                    candidate = word ^ other
                    if not candidate:
                        continue
                    weight = candidate.bit_count()
                    reading = min(reading, weight)
                    if weight <= reading + 4:
                        candidates.append(candidate)
                        if len(candidates) > 192:
                            candidates = sorted(set(candidates),
                                                key=int.bit_count)[:32]

            diagnostic.shuffle(basis)
            for word in light[:6]:
                weight = word.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word = candidate
                            weight = candidate.bit_count()
                            changed = True
                    if not changed:
                        break
                reading = min(reading, weight)
                candidates.append(word)

            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(basis, 3)
                    word = a ^ b ^ c
                    if word:
                        reading = min(reading, word.bit_count())
                        candidates.append(word)
            retained = sorted(set(retained + candidates),
                              key=int.bit_count)[:24]
            reading = min(reading, retained[0].bit_count())
            if reading <= 12:
                break
        return reading, retained

    def query_record(record, offset, trials):
        d, words = kernel_query(record["columns"], seed+offset, trials)
        record["d"] = min(record.get("d", d), d)
        record["words"] = sorted(
            set(record.get("words", []) + words),
            key=int.bit_count)[:24]

    def quality(record):
        return (record["d"], -record["cycles"][0],
                -record["cycles"][1], record["rank"])

    def select_component(side):
        records, signatures = [], set()
        wanted = 8 if side == "A" else 4
        count = 40 if side == "A" else 28
        offset = 100003 if side == "A" else 200003
        for batch in range(3):
            for _ in range(count):
                record = inspect(fresh(side), side)
                if record is None or record["signature"] in signatures:
                    continue
                signatures.add(record["signature"])
                records.append(record)
            records.sort(key=lambda r: (r["cycles"], -r["rank"]))
            screened = []
            for record in records:
                if not kernel4_free(record["columns"]):
                    continue
                query_record(record, offset, 10 if side == "A" else 8)
                screened.append(record)
                if len(screened) >= wanted:
                    break
            if screened:
                break
        if not screened:
            raise ValueError("no component passed verified kernel-four gates")

        screened.sort(key=quality, reverse=True)
        finalists = screened[:3 if side == "A" else 2]
        for record in finalists:
            query_record(record, offset+700001,
                         32 if side == "A" else 20)
        finalists.sort(key=quality, reverse=True)

        if side == "A":
            # Repair individual normal coordinates, not a global gauge.
            # Every accepted proposal gets an independent confirmation.
            current = finalists[0]
            positions = [(i,j,t) for i, row in enumerate(current["grid"])
                         for j, cell in enumerate(row)
                         for t in range(len(cell))]
            for move in range(12):
                grid = [[cell[:] for cell in row]
                        for row in current["grid"]]
                i, j, t = rng.choice(positions)
                old = grid[i][j][t]
                available = [(a, old[1]) for a in range(37)
                             if (a, old[1]) not in grid[i][j]]
                grid[i][j][t] = rng.choice(available)
                proposal = inspect(grid, side)
                if proposal is None:
                    continue
                if not kernel4_free(proposal["columns"]):
                    continue
                query_record(proposal, offset, 12)
                if quality(proposal) <= quality(current):
                    continue
                query_record(proposal, offset+700001, 32)
                if quality(proposal) > quality(current):
                    current = proposal
            if all(current["signature"] != r["signature"]
                   for r in finalists):
                finalists.append(current)
            finalists.sort(key=quality, reverse=True)
            finalists = finalists[:3]

        for record in finalists:
            # Separate longer searches can only lower retained readings.
            query_record(record, offset+1700009,
                         64 if side == "A" else 32)
        finalists.sort(key=quality, reverse=True)
        return finalists[:2]

    aa = select_component("A")
    bb = select_component("B")

    def span_pivots(rows):
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
        # Membership test only; no claim that this partial remainder
        # is a canonical linear quotient signature.
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def quantum_reading(hx, hz, a, b, n):
        """Verify sampled localized witnesses against original stabilizers."""
        zspan, xspan = span_pivots(hz), span_pivots(hx)
        reading = n+1
        for word in a["words"][:12]:
            for beta in range(5):
                actual = sum(((word >> (j*q)) & mask)
                             << ((j*5+beta)*q) for j in range(2))
                if (all((actual & row).bit_count() % 2 == 0 for row in hx)
                        and nontrivial(actual, zspan)):
                    reading = min(reading, actual.bit_count())
        for word in b["words"][:12]:
            for j in range(2):
                actual = word << (j*5*q)
                if (all((actual & row).bit_count() % 2 == 0 for row in hz)
                        and nontrivial(actual, xspan)):
                    reading = min(reading, actual.bit_count())
        return reading

    best = None
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1443 and k >= 222
                    and 125*n <= 1000*k < 175*n):
                continue
            dq = quantum_reading(hx, hz, a, b, n)
            detected = min(a["d"], b["d"], dq)
            # Component readings remain selection heuristics, even when
            # particular embedded quantum witnesses are verified.
            merit = (detected, k*detected*detected,
                     a["d"]+b["d"],
                     -a["cycles"][0]-b["cycles"][0],
                     -a["cycles"][1]-b["cycles"][1])
            if best is None or merit > best[0]:
                best = (merit, A, B, k, hx, hz,
                        (a["d"], b["d"], dq),
                        (a["cycles"], b["cycles"]))
    if best is None:
        raise ValueError("no independently screened pair retained cell 0.150")
    assert css_ok(best[4], best[5])
    assert max(row.bit_count() for row in best[4]+best[5]) <= min(12, W_MAX)
    return {"idx": idx, "cells": {"A": best[1], "B": best[2]},
            "card": (model, topology, cohort), "k": best[3],
            "screen": best[0], "witness_readings": best[6],
            "component_cycles": best[7]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete independently optimized cell without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent measurements: rolled positive-cell supports on Z19:Z3 (4,2)x(5,3) reliably avoid small sector-2 logicals; measured slots 2 and 9 both contain X logicals of weight 6.
# REFUTED: Mixed-coset labeling with small-kernel and four-cycle optimization alone removes the A bottleneck on this bed; all five measured minima remained Z-type weight 22..24 at both Acols within one Bcol, while measured X readings were 74..85.
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
