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
CONCEPT = ("For S3 x D14, order 84, shapes (1,3)x(2,5), group-spanning trinomials and verified component screening should raise the 0.350 cell's five-code fitness above 35.3; distance readings remain provisional.")
GROUP = ("Use the direct product of S3 and the order-14 dihedral group, with independent reflection actions and trivial centre. This is genuinely different from the parent's order-74 dihedral group. Successful nonabelian beds motivate the experiment but do not predict its distance. Fix q=84.")
SHAPES = ("A=(1,3), B=(2,5): n=17*84=1428, k>=6*84=504, guaranteed rate 6/17=0.352941 in cell 0.350. Components have lengths 252 and 420 with 84 and 168 checks. Parent measured minima were sector-2 Arow/Brow operators, including weight 20 in slot 46; forward unequal components replace that bottleneck.")
FILLING = ("Twenty explicit support designs give B-column degrees 3 or 4 and A-column degrees (3,3,3) or (2,3,3). The sole degree-two column uses a binomial ratio of order 21. Every larger cell contains relative labels generating all 84 group elements. Both quantum row-weight sums are <=12. Select fresh labels using verified component small-kernel counts, four-cycles and repeated circuit searches; enforce final CSS, exact ranks and rate [0.325,0.375). Component readings are heuristics, not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r035_s3xd14_13x25_spanning_triples_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # (C3 semidirect C2) x (C7 semidirect C2).
    # Encoding: (rotation3, reflection3, rotation7, reflection7).
    els = [(a, b, c, d) for d in range(2) for c in range(7)
           for b in range(2) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 2, g[2] % 7, g[3] % 2)

    def mul(g, h):
        return ((g[0] + (-1 if g[1] else 1)*h[0]) % 3,
                (g[1] + h[1]) % 2,
                (g[2] + (-1 if g[3] else 1)*h[2]) % 7,
                (g[3] + h[3]) % 2)

    def inv(g):
        return ((g[0] if g[1] else -g[0]) % 3, g[1],
                (g[2] if g[3] else -g[2]) % 7, g[3])

    def idx(g):
        a, b, c, d = red(g)
        return ((d*7+c)*2+b)*3+a

    return dict(elements=els, ident=(0, 0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty support hypotheses, optimized without submitting orbit copies."""
    # First twelve designs: A degrees 3,3,3; B column degrees all 3.
    # These B count multisets distinguish row sums and zero/triple counts.
    first_rows = (
        (1,1,1,1,2), (0,1,1,2,2), (0,0,2,2,2),
        (0,1,1,1,3), (0,0,1,2,3), (0,0,0,3,3),
        (1,1,1,2,2), (0,1,2,2,2), (1,1,1,1,3),
        (0,1,1,2,3), (0,0,2,2,3), (0,0,1,3,3),
    )
    plans = [((3,3,3), (3,3,3,3,3), row)
             for row in first_rows]
    # Eight degree-transfer experiments: A sum 8 permits B degree 4.
    plans += [
        ((2,3,3), (4,3,3,3,3), (2,1,1,2,2)),
        ((2,3,3), (4,3,3,3,3), (1,1,2,2,2)),
        ((2,3,3), (4,3,3,3,3), (0,2,2,2,2)),
        ((2,3,3), (4,4,3,3,3), (2,2,1,1,2)),
        ((2,3,3), (4,4,3,3,3), (1,3,1,1,2)),
        ((2,3,3), (4,4,3,3,3), (0,3,1,2,2)),
        ((2,3,3), (4,4,4,3,3), (2,2,2,1,2)),
        ((2,3,3), (4,4,4,3,3), (1,3,2,1,2)),
    ]
    adegrees, bdegrees, first = plans[idx % 20]
    sa = [list(adegrees)]
    sb = [list(first), [d-x for d, x in zip(bdegrees, first)]]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q = G.q

    def draw_cell(count):
        if count == 0:
            return []
        root = G.pick(rng)
        if count == 1:
            return [root]
        if count == 2:
            # This normal-subgroup ratio has exact order lcm(3,7)=21.
            ratio = (rng.randrange(1, 3), 0,
                     rng.randrange(1, 7), 0)
            return [root, G.mul(root, ratio)]

        while True:
            x, y = G.pick(rng), G.pick(rng)
            # Independent images in the reflection quotient C2 x C2.
            if not ((x[1]*y[3]) ^ (x[3]*y[1])):
                continue
            comm = G.mul(G.mul(x, y), G.mul(G.inv(x), G.inv(y)))
            if comm[0] == 0 or comm[2] == 0:
                continue
            # comm generates C3 x C7; quotient independence therefore
            # proves that x,y generate the entire order-84 group.
            cell = [root, G.mul(root, x), G.mul(root, y)]
            break
        while len(cell) < count:
            g = G.pick(rng)
            if g not in cell:
                cell.append(g)
        return cell

    def sample_grid(support):
        return [[draw_cell(count) for count in row] for row in support]

    def component_data(grid, side):
        # Project identity-probe matrices built by the verified helper.
        # A uses its forward left lift; B uses its forward right lift.
        unit = [[[G.e]]]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        if side == 0:
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = [row & mask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = [row & mask for row in hz]
        rank = gf2_rank(rows)
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        seen = {}
        cycles = 0
        nr = len(rows)
        for column in columns:
            checks = []
            value = column
            while value:
                bit = value & -value
                checks.append(bit.bit_length() - 1)
                value ^= bit
            for a in range(len(checks)):
                for b in range(a):
                    key = checks[a] * nr + checks[b]
                    old = seen.get(key, 0)
                    cycles += old
                    seen[key] = old + 1
        return {"grid": grid, "columns": columns,
                "rank": rank, "nchecks": nr, "c4": cycles}

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

    def circuit_reading(columns, trials):
        """An upper reading for a classical kernel, never a certificate."""
        best = len(columns) + 1
        order = list(range(len(columns)))
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome = columns[c]
                word = 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    pivot = pivots.get(p)
                    if pivot is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= pivot[0]
                    word ^= pivot[1]
                if not syndrome:
                    basis.append(word)
                    best = min(best, word.bit_count())
            if best <= 10:
                return best
            # Short fundamental circuits and their pairwise cancellations.
            light = sorted(basis, key=int.bit_count)[:24]
            for a in light:
                for b in basis:
                    if a != b:
                        best = min(best, (a ^ b).bit_count())
            if best <= 10:
                return best
        return best

    def component_pool(support, side):
        records = [
            component_data(sample_grid(support), side)
            for _ in range(36)
        ]
        # Prefer near-full component rank, without treating it as the
        # quantum rank or rejecting an entire support hypothesis.
        near_full = [
            r for r in records
            if r["rank"] >= r["nchecks"] - 6
        ]
        if near_full:
            records = near_full
        records.sort(key=lambda r: (r["c4"], -r["rank"]))
        records = records[:8]
        for record in records:
            record["small"] = small_kernels(record["columns"])
        records.sort(key=lambda r: (r["small"], r["c4"], -r["rank"]))
        records = records[:5]
        for record in records:
            record["d"] = circuit_reading(record["columns"], 6)

        def quality(record):
            # Exact small-null-word evidence precedes heuristic readings.
            return (tuple(-v for v in record["small"]),
                    record["d"], -record["c4"], record["rank"])

        records.sort(key=quality, reverse=True)
        finalists = records[:3]
        for record in finalists:
            record["d"] = min(
                record["d"], circuit_reading(record["columns"], 32))
        finalists.sort(key=quality, reverse=True)
        return finalists

    for attempt in range(2):
        aa = component_pool(sa, 0)
        bb = component_pool(sb, 1)
        pairs = [(a, b) for a in aa for b in bb]

        def pair_quality(ab):
            a, b = ab
            small = tuple(a["small"][t] + b["small"][t]
                          for t in range(4))
            # Additional harness cohorts change screening emphasis, not
            # group-automorphism or permutation images of existing fills.
            if (idx // 20) % 2:
                cycles = max(a["c4"]/len(a["columns"]),
                             b["c4"]/len(b["columns"]))
            else:
                cycles = (a["c4"]/len(a["columns"])
                          + b["c4"]/len(b["columns"]))
            return (tuple(-v for v in small),
                    min(a["d"], b["d"]), a["d"] + b["d"], -cycles)

        pairs.sort(key=pair_quality, reverse=True)
        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1428 and k >= 504
                    and 325*n <= 1000*k < 375*n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= 12
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "k": k, "component_readings": (a["d"], b["d"])}
    raise ValueError("screened pairs missed the 0.350 rate window")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: w=12 alone prevents weight-4 logicals; parent slots 0-5 all had w=12 and d2s=4.
# REFUTED: Preliminary distance readings are stable under longer searches; the current D74 slot 46 fell from d2s=23 to d30s=20.
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
