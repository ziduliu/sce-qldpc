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
CONCEPT = ("Dic11, order 44, shapes (2,5)x(2,6): noncommuting cubic fillings selected by independently repeated component and full CSS-logical searches should produce five measured distance readings >=12 in cell 0.350. This is falsifiable measurement guidance, not a distance certificate.")
GROUP = ("Jump from abelian Z2 x Z22 to the dicyclic group Dic11: x has order 22, y^2=x^11, and yxy^-1=x^-1. Every element outside <x> has order four, unlike dihedral reflections. This order-44 bed is absent from BEDS. Successful nonabelian centre-two beds motivate the experiment without establishing distance transfer; fixed q=44 retains n=1496.")
SHAPES = ("Keep A=(2,5), B=(2,6): n=34*44=1496, k>=44*3*4=528, guaranteed rate 12/34=0.352941 targets cell 0.350. Components have 220 and 264 variables with 88 checks each. The parent's five measured minima were weight ten, entirely sector-1 Acol/Bcol operators, frequently emphasizing Acol2/3. Exact full ranks retain [0.325,0.375).")
FILLING = ("Twenty cards cross 0..3 concentrated A columns with 0..4 concentrated B columns. Every column has three terms; A-row totals 7/8 and B-row totals 9/9 give HX<=11 and HZ<=12. Binomials mix order-22 rotation ratios and order-four dicyclic ratios; each trinomial's relative labels generate all 44 elements. Optimize verified component cycles, screen exact kernels through four, and independently deepen component witnesses. Final pairing is selected by genuine full CSS null words outside the opposite stabilizer span, with independent longer retests. These sampled readings remain upper bounds, not certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Dic11_25x26_cubic_css_coset_search_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11 = <x,y | x^22=1, y^2=x^11, yxy^-1=x^-1>.
    # Canonical element (a,b) means x^a y^b, b in {0,1}.
    if not valid_q(q):
        raise ValueError("Dic11 requires order 44")
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        carry, b = divmod(b, 2)
        return ((a + 11*carry) % 22, b)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 11*b*d) % 22,
                (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a+11) % 22, 1) if b else ((-a) % 22, 0)

    def idx(g):
        a, b = red(g)
        return 22*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty concentration designs with actual CSS logical-coset selection."""
    import itertools
    import math

    q = G.q
    card, cohort = idx % 20, idx // 20
    diagnostic_seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    units = (1, 3, 5, 7, 9, 13, 15, 17, 19, 21)

    def support(width, concentrated, total):
        options = [(0, 3) if j < concentrated else (1, 2)
                   for j in range(width)]
        choices = [v for v in itertools.product(*options)
                   if sum(v) == total]
        first = list(rng.choice(choices))
        return [first, [3-v for v in first]]

    sa = support(5, card % 4, 7)
    sb = support(6, card // 4, 9)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache native operations in an indexed group, not a separate lift.
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
    monomials = {}
    all_cells = [[[g] for g in range(q)]]
    for side in ("A", "B"):
        if side == "A":
            source, _, _ = lifted_matrices(probe, all_cells, [[[]]])
        else:
            _, source, _ = lifted_matrices(probe, [[[]]], all_cells)
        monomials[side] = [
            [(row >> (g*q)) & mask for row in source]
            for g in range(q)
        ]

    def columns_of(rows, width):
        cols = [0] * width
        for r, raw in enumerate(rows):
            while raw:
                bit = raw & -raw
                cols[bit.bit_length()-1] |= 1 << r
                raw ^= bit
        return cols

    def draw_cell(count, phase):
        if count == 0:
            return []
        root = G.pick(rng)
        if count == 1:
            return [root]
        rotation = (rng.choice(units), 0)
        outside = (rng.randrange(22), 1)
        if count == 2:
            # Rotation has exact order 22; outside has exact order four.
            fraction = (0.15, 0.30, 0.45)[phase % 3]
            ratio = outside if rng.random() < fraction else rotation
            return [root, G.mul(root, ratio)]
        assert count == 3
        # A primitive rotation plus any outside element generate Dic11.
        return [root, G.mul(root, rotation), G.mul(root, outside)]

    def fresh(sup, phase):
        return [[draw_cell(count, phase) for count in row]
                for row in sup]

    def inspect(grid, side, objective):
        """Assemble verified monomial blocks by GF2 linearity."""
        width = len(grid[0])*q
        blocks = monomials[side]
        rows = []
        for cells in grid:
            indexed = [[G.idx(g) for g in cell] for cell in cells]
            for r in range(q):
                row = 0
                for j, ids in enumerate(indexed):
                    block = 0
                    for g in ids:
                        block ^= blocks[g][r]
                    row |= block << (j*q)
                rows.append(row)
        cols = columns_of(rows, width)
        bad = sum(c == 0 for c in cols) + width-len(set(cols))
        uses = {}
        four = 0
        adjacency = [0] * len(rows)
        forced = 0
        for syndrome in cols:
            degree = syndrome.bit_count()
            forced += degree*(degree-1)*(degree-2)//6
            checks = []
            word = syndrome
            while word:
                bit = word & -word
                r = bit.bit_length()-1
                checks.append(r)
                adjacency[r] |= syndrome ^ bit
                word ^= bit
            for t, a in enumerate(checks):
                for b in checks[:t]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            word = neighbors & (-1 << (a+1))
            while word:
                bit = word & -word
                b = bit.bit_length()-1
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
                word ^= bit
        excess = max(0, triangles-forced)
        peak = max(x.bit_count() for x in adjacency)
        cost = 1000000*bad + (320+80*objective)*four
        cost += excess + objective*peak
        return {"grid": grid, "rows": rows, "columns": cols,
                "cost": cost, "cycles": (four, excess, peak)}

    def short_penalty(cols):
        """Exact component null-word collision screen through weight four."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return 1000000
        pairs = set()
        hits = 0
        for i, a in enumerate(cols):
            for b in cols[:i]:
                value = a ^ b
                hits += int(value in singles or value in pairs)
                pairs.add(value)
        return hits

    def kernel_reading(cols, local, trials):
        """Find genuine classical kernel words; return an upper reading."""
        order = list(range(len(cols)))
        best = len(cols)+1
        retained = []
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
            for c in order:
                value, word = cols[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    best = min(best, word.bit_count())
                    words.append(word)
            light = sorted(words, key=int.bit_count)[:16]
            for word in light:
                for other in words + retained:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())
            for word in light[:6]:
                for sweep in range(2):
                    changed = False
                    weight = word.bit_count()
                    for other in words + retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    best = min(best, weight)
                    if not changed:
                        break
            retained = sorted(set(retained+light),
                              key=int.bit_count)[:24]
            if best <= 6:
                break
        return best

    def component_pool(sup, side, phase):
        objective = (card + phase) % 4
        keepers = {}

        def signature(record):
            return tuple(tuple(sorted(G.idx(g) for g in cell))
                         for row in record["grid"] for cell in row)

        def remember(record):
            key = signature(record)
            if key in keepers:
                return
            keepers[key] = record
            if len(keepers) > 6:
                worst = max(keepers, key=lambda k: keepers[k]["cost"])
                del keepers[worst]

        starts = []
        for _ in range(32):
            record = inspect(fresh(sup, phase), side, objective)
            starts.append(record)
            remember(record)
        starts.sort(key=lambda r: r["cost"])
        occupied = [(i, j) for i, row in enumerate(sup)
                    for j, count in enumerate(row) if count]

        for current in starts[:2]:
            for step in range(144):
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                i, j = rng.choice(occupied)
                grid[i][j] = draw_cell(sup[i][j], phase)
                proposed = inspect(grid, side, objective)
                delta = proposed["cost"]-current["cost"]
                temperature = q*(8*(1-step/144.0)**2 + 0.15)
                if delta <= 0 or rng.random() < math.exp(-delta/temperature):
                    current = proposed
                    remember(proposed)

        records = list(keepers.values())
        seed = diagnostic_seed + (0 if side == "A" else 100003)
        for record in records:
            record["short"] = short_penalty(record["columns"])
            record["rank"] = gf2_rank(record["rows"])
            record["d"] = kernel_reading(
                record["columns"], random.Random(seed), 24)

        def merit(record):
            return (-record["short"], record["d"],
                    -record["cost"], record["rank"])

        records.sort(key=merit, reverse=True)
        finalists = records[:3]
        for record in finalists:
            record["d"] = min(record["d"], kernel_reading(
                record["columns"], random.Random(seed+700001), 80))
        finalists.sort(key=merit, reverse=True)
        return finalists

    def span_pivots(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def in_span(word, pivots):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return False
            word ^= old
        return True

    def logical_reading(cols, stabilizers, span, seed, trials):
        """Actual CSS logical witnesses, not merely classical null words.

        All elimination words are in the opposite-check kernel.
        Stabilizer descent preserves their logical coset. Membership
        in the stabilizer row span is checked explicitly.
        """
        local = random.Random(seed)
        order = list(range(len(cols)))
        weights = [s.bit_count() for s in stabilizers]
        best, witness = len(cols)+1, None

        def consider(word, descend=False):
            nonlocal best, witness
            if not word:
                return
            weight = word.bit_count()
            if not descend and weight >= best:
                return
            if in_span(word, span):
                return
            if descend or weight < best:
                for sweep in range(2):
                    changed = False
                    for s, sw in zip(stabilizers, weights):
                        if 2*(word & s).bit_count() > sw:
                            word ^= s
                            changed = True
                    if not changed:
                        break
            weight = word.bit_count()
            if weight < best:
                best, witness = weight, word

        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
            for c in order:
                value, word = cols[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    words.append(word)
                    consider(word)
            light = sorted(words, key=int.bit_count)[:12]
            for word in light[:8]:
                consider(word, True)
            for word in light:
                for other in words:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() < best:
                        consider(candidate)
            if best <= 6:
                break

        if witness is None:
            raise ValueError("positive-k CSS search found no logical witness")
        syndrome = 0
        word = witness
        while word:
            bit = word & -word
            syndrome ^= cols[bit.bit_length()-1]
            word ^= bit
        assert syndrome == 0 and not in_span(witness, span)
        return best, witness

    def pair_merit(pair):
        a, b = pair
        return (-(a["short"]+b["short"]),
                min(a["d"], b["d"]), a["d"]+b["d"],
                -a["cost"]-b["cost"])

    for attempt in range(2):
        phase = cohort + attempt
        aa = component_pool(sa, "A", phase)
        bb = component_pool(sb, "B", phase+1)
        pairs = sorted(((a, b) for a in aa for b in bb),
                       key=pair_merit, reverse=True)
        candidates = []
        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            indexed_A = [[[G.idx(g) for g in cell] for cell in row]
                         for row in A]
            indexed_B = [[[G.idx(g) for g in cell] for cell in row]
                         for row in B]
            hx, hz, n = lifted_matrices(probe, indexed_A, indexed_B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1496 and k >= 528
                    and 325*n <= 1000*k < 375*n):
                continue

            # Confirm the cached components against the final full lift.
            ar = []
            for i in range(2):
                for r in range(q):
                    source = hx[(i*6)*q+r]
                    ar.append(sum(
                        ((source >> (j*6*q)) & mask) << (j*q)
                        for j in range(5)))
            br = [row & ((1 << (6*q))-1) for row in hz[:2*q]]
            assert ar == a["rows"] and br == b["rows"]

            checks = (hz, hx)
            stabilizers = (hx, hz)
            cols = [columns_of(rows, n) for rows in checks]
            spans = [span_pivots(rows) for rows in stabilizers]
            readings, witnesses = [], []
            for t in range(2):
                d, word = logical_reading(
                    cols[t], stabilizers[t], spans[t],
                    diagnostic_seed+2000003*t, 4)
                readings.append(d)
                witnesses.append(word)
            candidates.append({
                "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
                "cols": cols, "spans": spans,
                "readings": readings, "witnesses": witnesses,
                "component_readings": (a["d"], b["d"]),
                "cycles": (a["cycles"], b["cycles"]),
                "component_merit": pair_merit((a, b))
            })
            if len(candidates) == 4:
                break

        if not candidates:
            continue

        def css_merit(record):
            return (min(record["readings"]), sum(record["readings"]),
                    record["component_merit"])

        candidates.sort(key=css_merit, reverse=True)
        finalists = candidates[:2]
        for record in finalists:
            stabilizers = (record["hx"], record["hz"])
            for t in range(2):
                # Independent deeper searches can only lower the reading.
                d, word = logical_reading(
                    record["cols"][t], stabilizers[t], record["spans"][t],
                    diagnostic_seed+900001+2000003*t, 24)
                if d < record["readings"][t]:
                    record["readings"][t] = d
                    record["witnesses"][t] = word
        finalists.sort(key=css_merit, reverse=True)
        chosen = finalists[0]
        A, B = chosen["A"], chosen["B"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert hx == chosen["hx"] and hz == chosen["hz"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert _grid_weights(sa, sb) <= W_MAX
        return {
            "idx": idx, "card": (card % 4, card // 4, phase),
            "cells": {"A": A, "B": B}, "k": chosen["k"],
            "component_readings": chosen["component_readings"],
            "css_logical_readings": tuple(chosen["readings"]),
            "component_cycles": chosen["cycles"]
        }

    raise ValueError("no independently screened dicyclic pair in cell 0.350")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled positive-cell supports with uniform labels suppress sector-1 weight-6 logicals on this bed; all five measured parent codes instead have d=6 and sector-1 minimum witnesses.
# REFUTED: Balanced abelian cubic columns and component weight-six screening lift the five measured distances above ten and provide a novel code per slot; all five current readings were ten, and 31 of 60 slots were isomorphic duplicates.
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
