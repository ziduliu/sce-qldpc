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
CONCEPT = ("For SL(2,5), order 120, A=(1,2), B=(2,5), group-spanning higher-weight cells and B-column degrees three or four should yield five measured distance readings >=16 in cell 0.250. This targets the weaker frontier incumbent rather than the parent's 0.300 plateau; readings remain provisional upper bounds.")
GROUP = ("Jump from C67:C3 to the determinant-one matrix group SL(2,F5), order 120, with centre {I,-I}. This is a different construction family and a genuinely different group, not a relabeling. The related order-120 centre-two nine-class bed reached fitness 64.8 on (1,2)x(3,4); this motivates, but does not establish, transfer to the untried shapes here.")
SHAPES = ("A=(1,2), B=(2,5): q=120 gives n=12*120=1440, k>=120*(2-1)*(5-2)=360 and guaranteed rate 0.250. Components have 240 variables/120 checks and 600 variables/240 checks. Parent measured minima were entirely sector-1 Acol/Bcol witnesses, prominently Acol1 across multiple B columns. Both components change; exact full ranks enforce 0.225<=k/n<0.275.")
FILLING = ("Twenty cards cross four density allocations with five B concentration budgets. A columns have degree three or four; every B column has degree three or four, removing the parent's degree-two B-column bottleneck. Binomial ratios have order five or ten. Larger cells exclude involutory ratios and have relative generators spanning all 120 elements. Verified component lifts screen kernels through four and short cycles; independent component and actual quantum witness searches select finalists. HX maxima are 10,11,11,12 and HZ maxima are all 12. Fresh independent fillings, not orbit copies, are submitted; none of these screens certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_SL2F5_12x25_spanning_column3_actual_logicals_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    # SL(2,F5), NOT its quotient PSL(2,F5).
    # Element (a,b,c,d) encodes the matrix (a,b;c,d).
    if not valid_q(q):
        raise ValueError("SL(2,F5) requires order 120")

    def red(g):
        return tuple(int(x) % 5 for x in g)

    els = [(a, b, c, d)
           for a in range(5) for b in range(5)
           for c in range(5) for d in range(5)
           if (a*d - b*c) % 5 == 1]
    assert len(els) == q
    pos = {g: i for i, g in enumerate(els)}

    # Cache the matrix law; all screening lifts still use the scaffold.
    table = [
        [((a*e + b*h) % 5, (a*f + b*k) % 5,
          (c*e + d*h) % 5, (c*f + d*k) % 5)
         for e, f, h, k in els]
        for a, b, c, d in els
    ]
    inverses = {
        g: (g[3], (-g[1]) % 5, (-g[2]) % 5, g[0])
        for g in els
    }

    def mul(g, h):
        return table[pos[red(g)]][pos[red(h)]]

    def inv(g):
        return inverses[red(g)]

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# (A cell counts, B column degrees, first B-row total).
# Cross each allocation with 0..4 concentrated B columns.
# HX maxima: 10,11,11,12. HZ maxima: 12,12,12,12.
_DENSITY_CARDS = (
    ((3, 3), (4, 4, 4, 3, 3), 9),
    ((3, 4), (4, 3, 3, 3, 3), 8),
    ((4, 4), (3, 3, 3, 3, 3), 7),
    ((4, 4), (4, 3, 3, 3, 3), 8),
)


def _sl5_support(rng, degrees, concentrated, first_total):
    """Exact B-column degrees and row margins, with a prescribed zero count."""
    import itertools
    options = [
        (0, d) if j < concentrated else tuple(range(1, d))
        for j, d in enumerate(degrees)
    ]
    choices = [
        p for p in itertools.product(*options)
        if sum(p) == first_total
    ]
    if not choices:
        raise ValueError("infeasible SL2 support card")
    first = list(rng.choice(choices))
    return [first, [d-x for d, x in zip(degrees, first)]]


def _sl5_spanning(G, generators):
    """Exact subgroup closure of the normalized within-cell labels."""
    reached = {G.e}
    queue = [G.e]
    for g in queue:
        for h in generators:
            gh = G.mul(g, h)
            if gh not in reached:
                reached.add(gh)
                queue.append(gh)
    return len(reached) == G.q


def _draw_sl5(rng, G, support, orders, cohort):
    """Fresh matrix labels, long binomial ratios and spanning larger cells."""
    # Orders five and ten give different binomial cycle lengths, not
    # merely different names for the same group elements.
    ratio_order = (5, 10)[cohort % 2]
    ratios = [g for g in G.elements if orders[g] == ratio_order]
    result = []
    for row in support:
        out = []
        for count in row:
            if count == 0:
                out.append([])
                continue
            if count == 1:
                out.append([G.pick(rng)])
                continue
            if count == 2:
                root = G.pick(rng)
                out.append([root, G.mul(root, rng.choice(ratios))])
                continue

            for _ in range(512):
                cell = rng.sample(G.elements, count)
                inverses = [G.inv(g) for g in cell]
                if any(
                    orders[G.mul(inverses[i], cell[j])] == 2
                    for i in range(count) for j in range(i)
                ):
                    continue
                generators = [
                    G.mul(inverses[0], g) for g in cell[1:]
                ]
                if not _sl5_spanning(G, generators):
                    continue
                out.append(cell)
                break
            else:
                raise ValueError("no spanning matrix support sampled")
        result.append(out)
    return result


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _kernel4_free(columns):
    """Exact component exclusion through weight four."""
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


def _component_rows(hx, hz, q, side):
    """Extract the actual left/right components from verified matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    mask = (1 << q) - 1
    rows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            rows.append(row)
    return rows


def _cycle_cost(rows, penalty):
    """Count component four/six cycles, excluding reused variable nodes."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    for a in range(h):
        ns = upper[a]
        for t, b in enumerate(ns):
            ab = common[a][b]
            for c in ns[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return penalty * c4 + c6


def _connected_component(rows, columns):
    """Connectedness of the actual component Tanner graph."""
    adjacency = [0] * len(rows)
    for syndrome in columns:
        word = syndrome
        while word:
            bit = word & -word
            r = bit.bit_length() - 1
            adjacency[r] |= syndrome ^ bit
            word ^= bit
    visited = frontier = 1
    while frontier:
        reached = 0
        word = frontier
        while word:
            bit = word & -word
            reached |= adjacency[bit.bit_length() - 1]
            word ^= bit
        frontier = reached & ~visited
        visited |= frontier
    return visited == (1 << len(rows)) - 1


def _span_pivots(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _span_signature(word, pivots):
    """Linear canonical remainder, including ALL missing-pivot bits."""
    remainder = 0
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            bit = 1 << p
            remainder |= bit
            word ^= bit
        else:
            word ^= old
    return remainder


def _kernel_probe(rng, columns, trials, signatures=None, stabilizers=()):
    """Sample verified kernel circuits and cancellations.

    With signatures, zero signature means membership in the ORIGINAL
    stabilizer span. Descents use only original stabilizer rows.
    The returned minimum is a sampled upper reading, never a certificate.
    """
    width = len(columns)
    best = width + 1
    best_word = None
    order = list(range(width))
    descent = list(stabilizers)

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        circuits = []
        for c in order:
            syndrome = columns[c]
            word = 1 << c
            tag = signatures[c] if signatures is not None else word
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word, tag)
                    break
                syndrome ^= old[0]
                word ^= old[1]
                tag ^= old[2]
            if not syndrome:
                circuits.append((word, tag))
                if tag and word.bit_count() < best:
                    best, best_word = word.bit_count(), word

        circuits.sort(key=lambda item: item[0].bit_count())
        light = circuits[:12]
        if signatures is not None:
            light += [item for item in circuits if item[1]][:4]
        for a, ta in light:
            for b, tb in circuits:
                if ta ^ tb:
                    candidate = a ^ b
                    weight = candidate.bit_count()
                    if weight < best:
                        best, best_word = weight, candidate

        if descent:
            rng.shuffle(descent)
            seeds = [word for word, tag in circuits if tag][:3]
            if best_word is not None:
                seeds.append(best_word)
            for word in seeds:
                for sweep in range(2):
                    changed = False
                    weight = word.bit_count()
                    for row in descent:
                        candidate = word ^ row
                        new_weight = candidate.bit_count()
                        if new_weight < weight:
                            word, weight = candidate, new_weight
                            changed = True
                    if not changed:
                        break
                if word.bit_count() < best:
                    best, best_word = word.bit_count(), word

    return best, best_word


def _component_pool(rng, G, support, side, orders, cohort):
    """Screen arbitrary-shaped components using only the verified lift."""
    unit = [[[G.e]]]
    width = len(support[0]) * G.q
    mask = (1 << width) - 1
    records = []

    for trial in range(40):
        grid = _draw_sl5(rng, G, support, orders, cohort)
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = [row & mask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = [row & mask for row in hz]
        columns = _columns(rows, width)
        if (0 in columns or len(set(columns)) != width
                or not _connected_component(rows, columns)):
            continue
        rank = gf2_rank(rows)
        if rank < len(rows) - 16:
            continue
        cost = _cycle_cost(rows, 160)
        records.append({"grid": grid, "rows": rows, "cols": columns,
                        "rank": rank, "cost": cost, "trial": trial})

    records.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))
    screened = []
    for record in records:
        if not _kernel4_free(record["cols"]):
            continue
        record["d"] = _kernel_probe(rng, record["cols"], 8)[0]
        screened.append(record)
        if len(screened) == 6:
            break
    screened.sort(key=lambda r: (r["d"], -r["cost"], r["rank"]),
                  reverse=True)

    finalists = screened[:2]
    for record in finalists:
        # Fresh elimination orders independently check an optimistic
        # preliminary component reading before pairing the components.
        record["d"] = min(record["d"],
                          _kernel_probe(rng, record["cols"], 32)[0])
    finalists.sort(key=lambda r: (r["d"], -r["cost"]), reverse=True)
    return finalists


def _quantum_probe(rng, hx, hz, n, trials):
    """Actual logical upper readings, not projected component homology."""
    reading = n + 1
    witnesses = []
    for opposite, stabilizers in ((hz, hx), (hx, hz)):
        columns = _columns(opposite, n)
        pivots = _span_pivots(stabilizers)
        signatures = [_span_signature(1 << c, pivots) for c in range(n)]
        d, word = _kernel_probe(rng, columns, trials,
                                signatures, stabilizers)
        if word is not None:
            # Check every retained witness in the original matrices.
            assert all((word & row).bit_count() % 2 == 0
                       for row in opposite)
            assert _span_signature(word, pivots) != 0
            witnesses.append(word.bit_count())
            reading = min(reading, d)
    return reading, tuple(witnesses)


def NEW_CTX(rng, G, idx):
    """Twenty density/concentration cards with independent witness confirmation."""
    counts_a, degrees_b, first_total = _DENSITY_CARDS[idx % 4]
    concentrated = (idx // 4) % 5
    cohort = idx // 20
    sa = [list(counts_a)]
    sb = _sl5_support(rng, degrees_b, concentrated, first_total)
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert min(sum(row[j] for row in sb)
               for j in range(B_SHAPE[1])) >= 3

    # SL(2,F5) element orders are at most ten. Compute them from the
    # supplied law rather than infer orders from matrix coordinates.
    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, 11):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("unexpected SL2 element order")

    # Explicit caps:
    # A=(3,3), B rows=(9,9), B max column=4: HX=10, HZ=12.
    # A=(3,4), B rows=(8,8), B max column=4: HX=11, HZ=12.
    # A=(4,4), B rows=(7,8), B max column=3: HX=11, HZ=12.
    # A=(4,4), B rows=(8,8), B max column=4: HX=12, HZ=12.
    for attempt in range(3):
        aa = _component_pool(rng, G, sa, "A", orders, cohort)
        bb = _component_pool(rng, G, sb, "B", orders, cohort)
        pairs = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 1440 and k >= 360
                        and 9 * n <= 40 * k < 11 * n):
                    continue
                merit = (min(a["d"], b["d"]), a["d"] + b["d"],
                         -a["cost"] - b["cost"])
                pairs.append({"A": A, "B": B, "hx": hx, "hz": hz,
                              "n": n, "k": k, "merit": merit,
                              "component_d": (a["d"], b["d"])})

        pairs.sort(key=lambda r: r["merit"], reverse=True)
        finalists = pairs[:2]
        if not finalists:
            continue
        for record in finalists:
            d, witnesses = _quantum_probe(
                rng, record["hx"], record["hz"], record["n"], 6)
            record["logical_d"] = d
            record["witnesses"] = witnesses

        # Confirm BOTH finalists using independent elimination orders.
        # These are actual logical witnesses modulo original stabilizers,
        # not a component-distance certificate.
        for record in finalists:
            d, witnesses = _quantum_probe(
                rng, record["hx"], record["hz"], record["n"], 12)
            record["logical_d"] = min(record["logical_d"], d)
            record["witnesses"] += witnesses

        def final_merit(record):
            da, db = record["component_d"]
            dn = record["logical_d"]
            return (min(da, db, dn), dn, min(da, db),
                    da + db, record["merit"][2], -(record["k"] - 360))

        chosen = max(finalists, key=final_merit)
        hx, hz = chosen["hx"], chosen["hz"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= 12
        return {"idx": idx, "card": (idx % 4, concentrated, cohort),
                "cells": {"A": chosen["A"], "B": chosen["B"]},
                "screen_k": chosen["k"],
                "component_readings": chosen["component_d"],
                "logical_reading": chosen["logical_d"],
                "verified_witness_weights": chosen["witnesses"]}
    raise ValueError("no screened filling passed the 0.250 exact-rank gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the independently screened full cell without new randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: Increasing A multiplicity while introducing a degree-two B column improves the C67:C3 (1,2)x(1,3) fillings: those density cards had d2s<=10, whereas measured all-trinomial cards reached 12 or 14 with sector-1 minima.
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
