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
CONCEPT = ("For metacyclic inversion Z29:Z2(s=28), order 58, shapes (1,2)x(2,4), independently optimized components with A weights (4,5) and degree-three B columns should raise five-code fitness above the parent's measured 45.")
GROUP = ("Keep the metacyclic inversion construction rule. The new size factor ten excludes order 118; order 58 is the largest admissible dihedral lift. This asymmetric bed already reached fitness 38.3, motivating a denser-A and independently optimized component test.")
SHAPES = ("Target cell 0.200: (1,2)x(2,4) gives n=580 and k>=116. A has 116 variables/58 checks; B has 232 variables/116 checks. The parent's measured weight-15 minima occupied sector-one Acol/Bcol blocks broadly. The new sector sizes are 464 and 116; verified k=116..130 keeps every retained code in [0.175,0.225).")
FILLING = ("Five prescribed support profiles crossed with four coset strategies define twenty optimization tasks. Four use A=(4,5), B column totals three and row totals at most seven; the fifth uses A=(4,4) and B column totals four. Both check-weight maxima are <=12. Optimize components independently using verified lifts, exact kernel screening through weight four, randomized dependency searches and four-cycle tie-breaks; combine finalists using exact ranks and stabilizer-tested sector-one witnesses. No search reading certifies distance.")
FAMILY_NAME = "r020f_m29r2s28_12x24_deg3_45_component_isd"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # Same metacyclic inversion construction, resized for factor ten.
    # Z_29 x| Z_2: y x y^-1 = x^28, with 28^2 = 1 mod 29.
    m, r, s = 29, 2, 28
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
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize two components, then test their actual CSS combinations."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask_q = (1 << q) - 1
    profiles = (
        ((4, 5), ((2,2,1,1), (1,1,2,2))),
        ((4, 5), ((3,2,1,0), (0,1,2,3))),
        ((4, 5), ((3,1,1,1), (0,2,2,2))),
        ((4, 5), ((2,2,2,1), (1,1,1,2))),
        ((4, 4), ((2,2,2,2), (2,2,2,2))),
    )
    aw, bw = profiles[idx % len(profiles)]
    supports = {"A": [list(aw)], "B": [list(row) for row in bw]}
    style = (idx // len(profiles)) % 4
    assert _grid_weights(supports["A"], supports["B"]) <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])
    pools = ([g for g in G.elements if g[1] == 0],
             [g for g in G.elements if g[1] == 1])

    def make_side(side):
        grid = []
        for i, row in enumerate(supports[side]):
            out = []
            for j, count in enumerate(row):
                phase = (i + j + idx // 20 + (side == "B")) & 1
                if not count:
                    cell = []
                elif style == 3:
                    cell = rng.sample(G.elements, count)
                elif count == 1:
                    cell = [rng.choice(pools[phase])]
                else:
                    if side == "B" and count == 2 and style in (0, 2):
                        # Non-involutory binomial differences within a coset.
                        nr = 2 * phase
                    elif style == 1:
                        nr = 1
                    elif style == 2 and side == "A":
                        nr = count - 1
                    else:
                        nr = count // 2 + (phase if count & 1 else 0)
                    cell = (rng.sample(pools[0], nr)
                            + rng.sample(pools[1], count - nr))
                out.append(cell)
            grid.append(out)
        return grid

    reference = {"A": make_side("A"), "B": make_side("B")}
    positions = {
        "A": [j * cb * q + h for j in range(ca) for h in range(q)],
        "B": list(range(cb * q)),
    }

    def restricted_columns(rows, side):
        pos = {p: c for c, p in enumerate(positions[side])}
        mask = sum(1 << p for p in positions[side])
        columns = [0] * len(pos)
        for r, row in enumerate(rows):
            x = row & mask
            while x:
                bit = x & -x
                columns[pos[bit.bit_length() - 1]] |= 1 << r
                x ^= bit
        return columns

    def component_columns(cells, side):
        A = cells if side == "A" else reference["A"]
        B = cells if side == "B" else reference["B"]
        hx, hz, _ = lifted_matrices(G, A, B)
        # Restriction removes the other component completely.
        return restricted_columns(hx if side == "A" else hz, side)

    def four_cycles(columns):
        pairs = {}
        for x in columns:
            neighbors = []
            while x:
                bit = x & -x
                neighbors.append(bit.bit_length() - 1)
                x ^= bit
            for a in range(len(neighbors)):
                for b in range(a):
                    key = (neighbors[b], neighbors[a])
                    pairs[key] = pairs.get(key, 0) + 1
        return sum(v * (v - 1) // 2 for v in pairs.values())

    def short_kernel(columns):
        """Exhaustive kernel screening through weight four, not logicals."""
        seen = {0: 0}
        for i, syndrome in enumerate(columns):
            word = 1 << i
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
        for i, syndrome in enumerate(columns):
            for j in range(i):
                s = syndrome ^ columns[j]
                word = (1 << i) | (1 << j)
                old = seen.get(s)
                if old is not None:
                    delta = old ^ word
                    if delta:
                        return delta.bit_count()
                else:
                    seen[s] = word
        return None

    def span_basis(rows):
        basis = {}
        for x in rows:
            while x:
                p = x.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = x
                    break
                x ^= old
        return basis

    def outside_span(x, basis):
        while x:
            p = x.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            x ^= old
        return False

    def witness_search(columns, trials, best, side=None, stabilizers=None):
        """Sample actual kernel words; optionally require CSS nontriviality."""
        order = list(range(len(columns)))

        def consider(word, bound):
            weight = word.bit_count()
            if not word or weight >= bound:
                return bound
            if stabilizers is None:
                return weight
            if side == "A":
                # A kernel word is a commuting Z operator in any B-column.
                for beta in range(cb):
                    full = 0
                    for j in range(ca):
                        full |= ((word >> (j*q)) & mask_q) << (
                            (j*cb + beta)*q)
                    if outside_span(full, stabilizers):
                        return weight
            else:
                # B kernel words are commuting X operators in any A-column.
                for j in range(ca):
                    if outside_span(word << (j*cb*q), stabilizers):
                        return weight
            return bound

        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            dependencies = []
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
                    dependencies.append(word)
                    best = consider(word, best)
            short = sorted(dependencies, key=int.bit_count)[:28]
            for a, word in enumerate(short):
                for other in short[:a]:
                    best = consider(word ^ other, best)
            # Greedy reductions remain in the verified component kernel.
            for seed in short[:4]:
                word, weight = seed, seed.bit_count()
                for _ in range(2):
                    changed = False
                    for other in dependencies:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if trial and tw < weight:
                            word, weight = trial, tw
                            changed = True
                    if not changed:
                        break
                best = consider(word, best)
            if best <= 4:
                break
        return best

    def component_key(candidate):
        return (int(candidate["short"] is None), candidate["d"],
                -candidate["deficit"], -candidate["cycles"])

    def optimize_side(side):
        targets = [(i, j) for i, row in enumerate(supports[side])
                   for j, count in enumerate(row) if count]
        keep = []
        quota = 32 if side == "A" else 48
        for attempt in range(quota):
            cells = make_side(side)
            columns = component_columns(cells, side)
            cycles = four_cycles(columns)
            # Preserve every cell's count and coset composition.
            for step in range(8):
                i, j = rng.choice(targets)
                cell = cells[i][j]
                t = rng.randrange(len(cell))
                old = cell[t]
                choices = [g for g in pools[old[1]] if g not in cell]
                cell[t] = rng.choice(choices)
                trial_columns = component_columns(cells, side)
                trial_cycles = four_cycles(trial_columns)
                if trial_cycles < cycles or (
                        trial_cycles == cycles and rng.random() < 0.5):
                    columns, cycles = trial_columns, trial_cycles
                else:
                    cell[t] = old

            short = short_kernel(columns)
            initial = short if short is not None else len(columns) + 1
            reading = witness_search(columns, 8, initial)
            checks = (ra if side == "A" else rb) * q
            candidate = {
                "cells": cells, "columns": columns,
                "cycles": cycles, "short": short, "d": reading,
                "deficit": checks - gf2_rank(columns),
            }
            keep.append(candidate)
            keep.sort(key=component_key, reverse=True)
            del keep[4:]

        # Equal deeper budgets prevent a lightly screened winner.
        for trials in (48, 96):
            for candidate in keep:
                candidate["d"] = witness_search(
                    candidate["columns"], trials, candidate["d"])
        keep.sort(key=component_key, reverse=True)
        return keep

    pa = optimize_side("A")
    pb = optimize_side("B")
    combinations = [(a, b) for a in pa for b in pb]
    combinations.sort(
        key=lambda pair: (
            min(pair[0]["d"], pair[1]["d"]),
            pair[0]["d"] + pair[1]["d"],
            -pair[0]["deficit"] - pair[1]["deficit"],
            -pair[0]["cycles"] - pair[1]["cycles"]),
        reverse=True)

    def improve(candidate, trials):
        candidate["da"] = witness_search(
            candidate["ac"], trials, candidate["da"],
            "A", candidate["zs"])
        candidate["db"] = witness_search(
            candidate["bc"], trials, candidate["db"],
            "B", candidate["xs"])

    def score(candidate):
        d = min(candidate["da"], candidate["db"])
        return (candidate["k"] * d * d / candidate["n"],
                candidate["da"] + candidate["db"], -candidate["cycles"])

    pool = []
    for a, b in combinations:
        A, B = a["cells"], b["cells"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 116 <= k <= 130:
            continue
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": a["columns"], "bc": b["columns"],
            "zs": span_basis(hz), "xs": span_basis(hx),
            "da": ca*q + 1, "db": cb*q + 1,
            "cycles": a["cycles"] + b["cycles"],
        }
        improve(candidate, 24)
        if candidate["da"] <= ca*q and candidate["db"] <= cb*q:
            pool.append(candidate)
        if len(pool) >= 6:
            break

    if not pool:
        raise ValueError("no component pair passed rank and logical screens")
    finalists = sorted(pool, key=score, reverse=True)[:4]
    for trials in (64, 128):
        for candidate in finalists:
            improve(candidate, trials)
    winner = max(finalists, key=score)
    assert css_ok(winner["hx"], winner["hz"])
    return {"idx": idx, "cells": {"A": winner["A"], "B": winner["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A legal row-weight budget alone prevents light logicals; parent slots 11, 24 and 46 have w=12 but d2s=2.
# REFUTED: Singleton-free mixed-coset filling and restricted-kernel screening eliminate the weight-15 sector-one plateau; all five measured current-parent leaders retained d30s=15 with sector-one minima.
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
