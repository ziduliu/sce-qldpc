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
CONCEPT = ("SL(2,5), order 120, A=(1,2), B=(3,4): four/five-term A cells without verified component 4-cycles, degree-three B columns, and systematic-basis kernel screening should produce five measured distance readings >=26 at rates below 0.125.")
GROUP = ("Frozen SL(2,5), q=120: this bed measured fitness 45.8416, with all five measured minima being sector-1 Z operators of weights 20..24. Keep the group and test whether the narrow A component, rather than the group, is the actionable bottleneck.")
SHAPES = ("Frozen (1,2)x(3,4): n=1320, k>=120, guaranteed rate=1/11 targets cell 0.100. Unequal component lengths 240 and 480 have 120 and 360 checks, respectively, and dimensions >=120. Parent minima mostly paired Acol0/Acol1 within individual Bcol blocks; slot 58 also had weight-20 witnesses confined to Acol1.")
FILLING = ("A has term counts (4,5); twenty non-row/column-equivalent B count profiles have column totals three and row totals <=7. HX weights are 9+3=12, and HZ weights are <=7+5=12. Construct A with distinct ordered quotients and verify its component has no 4-cycles. Screen connected components, exactly enumerate single-cell kernels of dimension <=14, and search actual kernel vectors by systematic-basis exchanges and pair combinations. Verified lifted ranks enforce rate<0.125. All sampled kernel readings are upper bounds, not quantum distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r010_SL2F5_12x34_sidon45_basis_exchange_v2"

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
    """Twenty count cards; all diagnostics use verified component rows."""
    from itertools import permutations, product

    q = G.q
    mask = (1 << q) - 1
    diagnostic_seed = rng.randrange(1 << 62)

    # Enumerate support experiments modulo protograph row/column
    # permutations, rather than spending cards on permuted templates.
    row_perms = tuple(permutations(range(3)))
    triples = ((1, 1, 1),) + tuple(sorted(set(
        permutations((2, 1, 0)))))
    profiles = set()
    for columns in product(triples, repeat=4):
        totals = [sum(c[r] for c in columns) for r in range(3)]
        if min(totals) < 1 or max(totals) > 7:
            continue
        canonical = min(
            tuple(sorted(tuple(c[r] for r in p) for c in columns))
            for p in row_perms)
        profiles.add(canonical)

    def profile_key(columns):
        totals = [sum(c[r] for c in columns) for r in range(3)]
        zeros = sum(x == 0 for c in columns for x in c)
        return (max(totals) - min(totals), zeros, columns)

    profiles = sorted(profiles, key=profile_key)
    if len(profiles) < 20:
        raise ValueError("insufficient inequivalent support cards")
    bc = profiles[idx % 20]
    sa = [[4, 5]]
    sb = [[bc[j][i] for j in range(4)] for i in range(3)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def make_A():
        """Distinct ordered quotients across both cells: a Sidon rule.
        The resulting 4-cycle assertion is checked on the verified lift.
        No automorphism or gauge images are used to generate slots.
        """
        cells = [[], []]
        quotients = set()
        for side, count in enumerate((4, 5)):
            for _ in range(count):
                for trial in range(400):
                    g = G.pick(rng)
                    gi = G.inv(g)
                    added = []
                    for h in cells[side]:
                        added.append(G.idx(G.mul(g, G.inv(h))))
                        added.append(G.idx(G.mul(h, gi)))
                    fresh = set(added)
                    if len(fresh) != len(added):
                        continue
                    if fresh & quotients:
                        continue
                    cells[side].append(g)
                    quotients.update(fresh)
                    break
                else:
                    return None
        return [cells]

    def kernel_basis(rows, width):
        """Actual kernel basis, systematic on its dependency columns."""
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        pivots = {}
        basis, information = [], []
        for c, source in enumerate(columns):
            syndrome = source
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
                information.append(c)
        return basis, information

    def exchange_reading(basis, information, width, steps, seed):
        """Information-set walk; every tested word is in the kernel.
        Exchanging one systematic coordinate preserves a complete basis.
        Inspect basis rows and pair combinations, not just one basis.
        """
        if not basis:
            return width + 1
        diagnostic = random.Random(seed)
        words = list(basis)
        coordinates = list(information)
        occupied = set(coordinates)
        reading = min(word.bit_count() for word in words)

        def pair_reading():
            light = sorted(range(len(words)),
                           key=lambda a: words[a].bit_count())[:24]
            result = width + 1
            for a in light:
                x = words[a]
                for b, y in enumerate(words):
                    if a != b:
                        result = min(result, (x ^ y).bit_count())
            return result

        reading = min(reading, pair_reading())
        for step in range(steps):
            c = diagnostic.randrange(width)
            if c in occupied:
                continue
            bit = 1 << c
            carriers = [a for a, word in enumerate(words) if word & bit]
            if not carriers:
                continue
            a = diagnostic.choice(carriers)
            pivot_word = words[a]
            for b in carriers:
                if b != a:
                    words[b] ^= pivot_word
                    reading = min(reading, words[b].bit_count())
            occupied.remove(coordinates[a])
            occupied.add(c)
            coordinates[a] = c
            if step % 64 == 63:
                reading = min(reading, pair_reading())
            # Already-observed short words cannot be screened away.
            if reading <= 18:
                break
        return reading

    def single_cell_reading(rows, seed):
        basis, information = kernel_basis(rows, q)
        if not basis:
            return q + 1, True
        if len(basis) <= 14:
            # Gray-code enumeration is exact for this small kernel.
            word = 0
            reading = q + 1
            for t in range(1, 1 << len(basis)):
                coordinate = (t & -t).bit_length() - 1
                word ^= basis[coordinate]
                reading = min(reading, word.bit_count())
            return reading, False
        return exchange_reading(
            basis, information, q, 512, seed), False

    keepers = []
    for attempt in range(72):
        A = make_A()
        if A is None:
            continue
        B = [[rng.sample(G.elements, count) for count in row]
             for row in sb]
        hx, hz, n = lifted_matrices(G, A, B)
        ha = [(row & mask) | (((row >> (4*q)) & mask) << q)
              for row in hx[:q]]
        hb = [row & ((1 << (4*q)) - 1) for row in hz[:3*q]]
        seed = diagnostic_seed + 10007 * attempt
        diagnostic = random.Random(seed)

        _, ca4, oka = _component_reading(diagnostic, ha, 2*q, 0)
        _, cb4, okb = _component_reading(diagnostic, hb, 4*q, 0)
        if not oka or not okb or ca4 != 0:
            continue

        local4, _ = single_cell_reading(
            [row & mask for row in ha], seed + 1)
        local5, unit5 = single_cell_reading(
            [(row >> q) & mask for row in ha], seed + 2)
        local_cap = min(local4, local5)

        abasis, ainfo = kernel_basis(ha, 2*q)
        bbasis, binfo = kernel_basis(hb, 4*q)
        da = min(local_cap, exchange_reading(
            abasis, ainfo, 2*q, 128, seed + 3))
        db = exchange_reading(
            bbasis, binfo, 4*q, 128, seed + 4)
        key = (min(da, db), -cb4, da + db, int(unit5))
        if len(keepers) == 8 and key <= keepers[-1][0]:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8*k < n):
            continue
        record = {
            "A": A, "B": B, "hx": hx, "hz": hz,
            "k": k, "seed": seed, "da": da, "db": db,
            "abasis": abasis, "ainfo": ainfo,
            "bbasis": bbasis, "binfo": binfo,
            "local_cap": local_cap, "cb4": cb4, "unit5": unit5,
        }
        keepers.append((key, record))
        keepers.sort(key=lambda item: item[0], reverse=True)
        del keepers[8:]

    if not keepers:
        raise ValueError("no connected Sidon filling passed target-rate gate")

    # Stronger, independently seeded readings on the shortlisted designs.
    # Earlier low readings are retained; additional search cannot erase
    # an already-found short kernel vector.
    best, best_key = None, None
    steps = 4096 + 512 * ((idx // 20) % 3)
    for _, record in keepers:
        seed = record["seed"] + 1000003
        da = min(record["da"], record["local_cap"], exchange_reading(
            record["abasis"], record["ainfo"], 2*q, steps, seed))
        db = min(record["db"], exchange_reading(
            record["bbasis"], record["binfo"], 4*q, steps, seed + 1))
        key = (min(da, db), -record["cb4"], da + db,
               int(record["unit5"]), -(record["k"] - q))
        if best_key is None or key > best_key:
            best_key, best = key, record

    if not css_ok(best["hx"], best["hz"]):
        raise ValueError("verified construction failed CSS check")
    return {"idx": idx, "card": idx % 20,
            "cells": {"A": best["A"], "B": best["B"]},
            "screen_k": best["k"], "diagnostic": best_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell, with no further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Connected-component screening with four/two elimination orders removes all sub-24 sector-1 logicals; parent measured slots 39, 48 and 58 instead had Z minima 22, 21 and 20, with slot 58 confined to Acol1.
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
