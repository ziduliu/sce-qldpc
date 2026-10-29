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
CONCEPT = ("SL(2,5), order 120, A=(1,2), B=(3,4): four-cycle-free A supports, including new (4,5)-term profiles, should raise the five-code mean distance reading above the parent's 22.2 while keeping mean rate below 0.125.")
GROUP = ("Keep the frozen SL(2,5) bed, measured at fitness 45.8416. Its unique nonidentity involution does not itself prevent short logicals; the parent produced screened slots with distance upper readings of eight.")
SHAPES = ("Keep A=(1,2), B=(3,4): n=1320, k>=120 and guaranteed rate=1/11, targeting cell 0.100. Components have lengths 240 and 480 with 120 and 360 checks. Parent minima were sector-1 Z witnesses at weights 20--24, usually spanning both A columns; slot 58 also had Acol1-only witnesses.")
FILLING = ("Twenty permutation-canonical B count designs are paired with A counts (4,5) or (3,5). Unique ordered relatives across A cells eliminate A-component four-cycles; higher-order relatives are preferred without forcing a subgroup. Screen 64 fillings, then deepen kernel searches on eight finalists using randomized elimination and kernel-vector reductions. Protect both components and enforce exact lifted k/n<0.125. Kernel readings are upper bounds on component distances, not quantum distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r010_SL2F5_12x34_sidon_robust_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL(2,5) has order 120")
    els = [(a,b,c,d) for a in range(5) for b in range(5)
           for c in range(5) for d in range(5)
           if (a*d-b*c) % 5 == 1]
    pos = {g:i for i,g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 5 for x in g)

    def mul(g,h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u)%5, (a*f+b*v)%5,
                (c*e+d*u)%5, (c*f+d*v)%5)

    def inv(g):
        a,b,c,d = g
        return (d%5, (-b)%5, (-c)%5, a%5)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


def _component_reading(rng, rows, nc, trials):
    """Connectedness, four-cycles, and sampled classical kernel weights.
    Random column elimination produces actual kernel vectors, hence
    upper bounds only; this is not a quantum distance certificate."""
    columns = [0] * nc
    parent = list(range(nc))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = {}
    cycles = 0
    for r, row in enumerate(rows):
        support = []
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            support.append(c)
            columns[c] |= 1 << r
            row ^= bit
        if support:
            anchor = root(support[0])
            for c in support[1:]:
                parent[root(c)] = anchor
        for p, c in enumerate(support):
            for d in support[p+1:]:
                key = (c, d)
                previous = pairs.get(key, 0)
                cycles += previous
                pairs[key] = previous + 1

    connected = (all(columns)
                 and len({root(c) for c in range(nc)}) == 1)
    if not connected:
        return 0, cycles, False

    reading = nc + 1
    for _ in range(trials):
        order = list(range(nc))
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome = columns[c]
            vector = 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, vector)
                    break
                syndrome ^= old[0]
                vector ^= old[1]
            if not syndrome:
                reading = min(reading, _popcount(vector))
    return reading, cycles, True


def NEW_CTX(rng, G, idx):
    """Canonical density experiments with Sidon A labels and deeper screening.
    All matrix diagnostics are extracted from the verified LP construction."""
    from itertools import product, permutations, combinations_with_replacement

    def count_profiles(column_total):
        triples = [t for t in product(range(3), repeat=3)
                   if sum(t) == column_total]
        row_perms = list(permutations(range(3)))
        representatives = set()
        for columns in combinations_with_replacement(triples, 4):
            totals = [sum(c[i] for c in columns) for i in range(3)]
            if not all(3 <= t <= 7 for t in totals):
                continue
            if not all(sum(c[i] != 0 for c in columns) >= 2
                       for i in range(3)):
                continue
            canonical = min(
                tuple(sorted(tuple(c[i] for i in p) for c in columns))
                for p in row_perms)
            representatives.add(canonical)

        def density_key(columns):
            totals = [sum(c[i] for c in columns) for i in range(3)]
            return (sum(x == 0 for c in columns for x in c),
                    sum(t*t for t in totals), columns)

        return sorted(representatives, key=density_key)

    # Interleave two genuinely different A densities. Canonicalizing B
    # counts prevents spending the twenty designs on support permutations.
    families = [((4, 5), count_profiles(3)),
                ((3, 5), count_profiles(4))]
    plans = []
    for t in range(max(len(p) for _, p in families)):
        for ac, profiles in families:
            if t < len(profiles):
                plans.append((ac, profiles[t]))
    if len(plans) < 20:
        raise ValueError("fewer than twenty canonical density experiments")
    ac, bc = plans[idx % 20]
    sa = [list(ac)]
    sb = [[bc[j][i] for j in range(4)] for i in range(3)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    mask = (1 << q) - 1
    elements = G.elements
    inverses = [G.inv(g) for g in elements]
    orders = []
    for g in elements:
        h = G.e
        for exponent in range(1, q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                orders.append(exponent)
                break
        else:
            raise ValueError("invalid element order")

    def relatives(g, h):
        return (G.idx(G.mul(g, inverses[G.idx(h)])),
                G.idx(G.mul(h, inverses[G.idx(g)])))

    def sidon_A():
        # An A-column gauge fixes its first label to the identity.
        # Distinct g*h^-1 over BOTH cells exclude A-row four-cycles.
        for restart in range(40):
            used = set()
            cells = []
            failed = False
            for count in ac:
                cell = [G.e]
                while len(cell) < count:
                    choices = []
                    for g in rng.sample(elements, 32):
                        if any(G._key(g) == G._key(h) for h in cell):
                            continue
                        differences = []
                        for h in cell:
                            differences.extend(relatives(g, h))
                        if len(set(differences)) != len(differences):
                            continue
                        if used.intersection(differences):
                            continue
                        # Soft preference only: connectivity is checked
                        # later, so a high-order cyclic subgroup cannot pass.
                        penalty = sum(3 if orders[d] == 3 else
                                      1 if orders[d] == 4 else 0
                                      for d in differences)
                        choices.append((penalty, rng.random(),
                                        g, differences))
                    if not choices:
                        failed = True
                        break
                    choices.sort(key=lambda item: item[:2])
                    # Keep label diversity rather than always taking one
                    # greedily determined high-order clique.
                    choice = choices[rng.randrange(min(3, len(choices)))]
                    cell.append(choice[2])
                    used.update(choice[3])
                if failed:
                    break
                cells.append(cell)
            if not failed:
                return [cells]
        return None

    def B_cell(count):
        if count < 2:
            return rng.sample(elements, count)
        # Here B counts are at most two. Avoid a relative involution,
        # which would create a two-term-cell four-cycle.
        while True:
            cell = rng.sample(elements, count)
            a, b = relatives(cell[0], cell[1])
            if a != b:
                return cell

    def components(hx, hz):
        ha = [(row & mask) | (((row >> (4*q)) & mask) << q)
              for row in hx[:q]]
        hb = [row & ((1 << (4*q)) - 1) for row in hz[:3*q]]
        return ha, hb

    def kernel_reading(rows, width, diagnostic, trials):
        """Actual sampled kernel words and their reductions; upper bound only."""
        columns = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                columns[bit.bit_length() - 1] |= 1 << r
                source ^= bit

        reading = width + 1
        for _ in range(trials):
            permutation = list(range(width))
            diagnostic.shuffle(permutation)
            pivots = {}
            basis = []
            for c in permutation:
                syndrome = columns[c]
                word = 1 << c
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

            # Every XOR below stays in the same component kernel.
            pool = sorted(basis, key=int.bit_count)[:24]
            for a, word in enumerate(pool):
                for other in pool[:a]:
                    reading = min(reading, (word ^ other).bit_count())
            anchors = pool[:8]
            for original in basis:
                word = original
                for _ in range(2):
                    before = word.bit_count()
                    for anchor in anchors:
                        changed = word ^ anchor
                        weight = changed.bit_count()
                        if 0 < weight < word.bit_count():
                            word = changed
                    if word.bit_count() == before:
                        break
                reading = min(reading, word.bit_count())
        return reading

    diagnostic_seed = rng.randrange(1 << 62)
    finalists = []
    for attempt in range(64):
        A = sidon_A()
        if A is None:
            continue
        B = [[B_cell(count) for count in row] for row in sb]
        hx, hz, n = lifted_matrices(G, A, B)
        ha, hb = components(hx, hz)
        diagnostic = random.Random(diagnostic_seed + 1009*attempt)
        da, ca4, oka = _component_reading(diagnostic, ha, 2*q, 6)
        if not oka:
            continue
        # The relative-element constraint is checked against the actual lift.
        if ca4 != 0:
            raise ValueError("Sidon A failed verified four-cycle check")
        db, cb4, okb = _component_reading(diagnostic, hb, 4*q, 2)
        if not okb:
            continue
        preliminary = (min(da, db), -cb4, da + db)
        finalists.append((preliminary, attempt, A, B,
                          hx, hz, ha, hb, cb4))
        finalists.sort(key=lambda item: item[0], reverse=True)
        del finalists[8:]

    best = None
    best_key = None
    for _, attempt, A, B, hx, hz, ha, hb, cycles in finalists:
        n = 11*q
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8*k < n):
            continue
        diagnostic = random.Random(diagnostic_seed + 1000003
                                  + 7919*attempt)
        da = kernel_reading(ha, 2*q, diagnostic, 64)
        db = kernel_reading(hb, 4*q, diagnostic, 8)
        key = (min(da, db), da + db, -cycles, k)
        if best_key is None or key > best_key:
            best_key = key
            best = {"A": A, "B": B}

    if best is None:
        raise ValueError("no connected filling passed exact target-rate gate")
    return {"idx": idx, "cells": best, "diagnostic": best_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell, with no further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Connected components alone exclude sub-12 logical witnesses: parent slots 29 and 32 had d2s=8 and slot 26 had d2s=10 despite passing the connectivity gate.
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
