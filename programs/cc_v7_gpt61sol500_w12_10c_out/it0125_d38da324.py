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
CONCEPT = ("Metacyclic Z1:Z23, s=1, order 23, shapes (3,4)x(3,4): longer components with nonzero rank-one augmentation columns and joint logical screening should exceed the parent's 22.9390 fitness while retaining k/n<0.125.")
GROUP = ("Keep the exact metacyclic group, order and encoding. At the new shapes q=23 gives n=575, so no order change is needed. The measured Z23 (2,3)x(2,3) parent reached 22.9390; this untried shape isolates a component-geometry experiment.")
SHAPES = ("Equal (3,4) sides give two 69-check/92-variable components, n=575 and guaranteed k>=23, rate 1/25. Rank-one augmentation designs can add trivial-character homology without forcing a weight-23 single-column norm kernel. Parent weight-14/16 minima mostly occupied sector-1 Acol/Bcol blocks; slot 20 also had a mixed-sector weight-14 X logical. Exact ranks gate every emitted rate below 0.125.")
FILLING = ("Sixteen cards cross two rank-one support geometries with four trinomial-ratio architectures; four further cards test even-support controls. All component rows contain six terms and columns contain four or six, giving both check-weight maxima twelve. Optimize independent labels using verified ranks, component witnesses and stabilizer-tested full-code logical witnesses. Independently deepen finalists and require different permutation-invariant uncolored Tanner fingerprints across emitted slots. Search readings remain heuristics, not distance certificates.")
FAMILY_NAME = "r010_m1r23s1_34x34_rankone_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 4)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)

# Every row total is six; column totals are a permutation of 6,4,4,4.
# The first two augmentation matrices have four identical nonzero
# columns (1,1,0)^T. The last profile has augmentation rank zero.
_SUPPORT_PROFILES = (
    ((3, 1, 1, 1), (3, 1, 1, 1), (0, 2, 2, 2)),
    ((3, 1, 1, 1), (1, 3, 1, 1), (0, 2, 2, 2)),
    ((0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 0, 2)),
)

# For triples {b,b+d,b+t*d}, t=2,3,4,5 represent four different
# affine triple classes over F23, not four automorphism images.
_TRIPLE_CARDS = (
    ((2, 2), (3, 3)),
    ((2, 3), (4, 4)),
    ((3, 4), (5, 5)),
    ((4, 5), (2, 5)),
)
_DESIGN_CARDS = tuple(
    (a, b, model)
    for a in range(2) for b in range(2) for model in range(4)
) + ((2, 2, 0), (2, 0, 1), (0, 2, 2), (2, 1, 3))


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _component_probe(cols, rng, trials=56):
    """Find classical kernel witnesses; absence of small ones is not proof."""
    width = len(cols)
    best = width + 1
    for _ in range(trials):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        deps = []
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, support)
                    break
                old_syndrome, old_support = pivots[p]
                syndrome ^= old_syndrome
                support ^= old_support
            if not syndrome:
                deps.append(support)
                best = min(best, support.bit_count())

        # Elementary basis operations preserve nonzero kernel witnesses.
        for _ in range(3):
            changed = False
            for a in range(len(deps)):
                v = deps[a]
                weight = v.bit_count()
                for b in range(len(deps)):
                    if a == b:
                        continue
                    z = v ^ deps[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                deps[a] = v
                best = min(best, weight)
            if not changed:
                break
    return best


def _row_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _outside_span(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _logical_reading(cols, stabilizers, rng, trials):
    """Actual full-code logical witnesses, including both LP sectors.

    Dependencies use columns of a verified matrix. A word contributes
    only when it is outside the opposite stabilizer row space.
    """
    n = len(cols)
    basis = _row_basis(stabilizers)
    best = n + 1
    order = list(range(n))
    memory = []
    for _ in range(trials):
        rng.shuffle(order)
        pivots, deps = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                deps.append(word)

        deps.sort(key=int.bit_count)
        seeds = [v for v in deps if _outside_span(v, basis)][:12]
        seeds.extend(memory)
        reducers = list(stabilizers) + deps[:64] + memory
        discoveries = []
        for seed in seeds:
            word, weight = seed, seed.bit_count()
            best = min(best, weight)
            for sweep in range(4):
                changed = False
                rng.shuffle(reducers)
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if (0 < tw < weight
                            and _outside_span(trial, basis)):
                        word, weight = trial, tw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
            discoveries.append(word)
        memory = sorted(set(discoveries), key=int.bit_count)[:12]
        if best <= 12:
            break
    return best


def _tanner_fingerprint(hx, hz, n):
    """Invariant of the uncolored union Tanner graph.

    Different fingerprints prove nonisomorphism even when check types
    are forgotten; equal fingerprints are conservatively rejected.
    """
    cols = _component_columns(hx + hz, n)
    pairs = {}
    for c, syndrome in enumerate(cols):
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pairs.setdefault((b, a), []).append(c)
    participation = [0] * n
    multiplicities = {}
    for vertices in pairs.values():
        size = len(vertices)
        multiplicities[size] = multiplicities.get(size, 0) + 1
        if size > 1:
            for c in vertices:
                participation[c] += size - 1
    return (
        tuple(sorted(row.bit_count() for row in hx + hz)),
        tuple(sorted((s.bit_count(), p)
                     for s, p in zip(cols, participation))),
        tuple(sorted(multiplicities.items())),
    )


def _draw_rankone_grid(rng, G, profile, triple_types):
    grid = []
    for i, counts in enumerate(profile):
        row = []
        for count in counts:
            if count == 3:
                t = triple_types[i]
                anchor = rng.randrange(G.q)
                delta = rng.randrange(1, G.q)
                cell = [(0, anchor),
                        (0, (anchor + delta) % G.q),
                        (0, (anchor + t * delta) % G.q)]
            else:
                cell = rng.sample(G.elements, count)
            assert len(cell) == len(set(cell))
            row.append(cell)
        grid.append(row)
    return grid


from functools import lru_cache


@lru_cache(maxsize=1)
def _design_bank(G):
    """Pure deterministic design optimization, shared across indexed slots."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    fingerprints = set()
    bank = []

    for slot in range(N_SLOTS):
        pa, pb, model = _DESIGN_CARDS[slot % len(_DESIGN_CARDS)]
        sa, sb = _SUPPORT_PROFILES[pa], _SUPPORT_PROFILES[pb]
        at, bt = _TRIPLE_CARDS[model]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED * 10009 + 340034 * slot + 231)
        selected = None

        def merit(item):
            d = min(item["da"], item["db"], item["full"], 32)
            return (item["k"] * d * d, d,
                    min(item["da"], 40) + min(item["db"], 40),
                    -max(item["cycles"]), -sum(item["cycles"]))

        # Extra batches are used only if all finalist fingerprints have
        # already appeared. No duplicate code is substituted as fallback.
        for batch in range(3):
            raw = []
            for attempt in range(192):
                A = _draw_rankone_grid(rng, G, sa, at)
                B = _draw_rankone_grid(rng, G, sb, bt)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (0 < k and 8 * k < n):
                    continue

                # Extract the actual component actions from verified HX/HZ.
                ar = []
                for i in range(ra):
                    for r in range(q):
                        source = hx[(i * cb) * q + r]
                        ar.append(sum(
                            ((source >> (j * cb * q)) & mask) << (j * q)
                            for j in range(ca)))
                br = [row & ((1 << (cb * q)) - 1)
                      for row in hz[:rb * q]]
                ac = _component_columns(ar, ca * q)
                bc = _component_columns(br, cb * q)
                cycles = (_component_four_cycles(ac),
                          _component_four_cycles(bc))
                raw.append({
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "cycles": cycles, "full": n + 1,
                })
                if len(raw) >= 48:
                    break

            raw.sort(key=lambda item: (
                max(item["cycles"]), sum(item["cycles"]), -item["k"]))
            pool = raw[:10]
            for item in pool:
                item["da"] = _component_probe(item["ac"], rng, 20)
                item["db"] = _component_probe(item["bc"], rng, 20)
                # Even-column controls have explicit weight-q norm words.
                if pa == 2:
                    item["da"] = min(item["da"], q)
                if pb == 2:
                    item["db"] = min(item["db"], q)

            finalists = sorted(pool, key=merit, reverse=True)[:4]
            for item in finalists:
                item["da"] = min(
                    item["da"], _component_probe(item["ac"], rng, 84))
                item["db"] = min(
                    item["db"], _component_probe(item["bc"], rng, 84))
                item["xc"] = _component_columns(item["hz"], item["n"])
                item["zc"] = _component_columns(item["hx"], item["n"])
                dx = _logical_reading(item["xc"], item["hx"], rng, 6)
                dz = _logical_reading(item["zc"], item["hz"], rng, 6)
                item["full"] = min(dx, dz)

            leaders = sorted(finalists, key=merit, reverse=True)[:3]
            for item in leaders:
                # Fresh searches can only lower retained readings.
                item["da"] = min(
                    item["da"], _component_probe(item["ac"], rng, 84))
                item["db"] = min(
                    item["db"], _component_probe(item["bc"], rng, 84))
                item["full"] = min(
                    item["full"],
                    _logical_reading(item["xc"], item["hx"], rng, 12),
                    _logical_reading(item["zc"], item["hz"], rng, 12))

            for item in sorted(leaders, key=merit, reverse=True):
                fingerprint = _tanner_fingerprint(
                    item["hx"], item["hz"], item["n"])
                if fingerprint in fingerprints:
                    continue
                selected = (item, fingerprint)
                break
            if selected is not None:
                break

        if selected is None:
            bank.append(None)
            continue
        item, fingerprint = selected
        assert css_ok(item["hx"], item["hz"])
        assert 100 <= item["n"] < 600
        assert 0 < item["k"] and 8 * item["k"] < item["n"]
        assert _grid_weights(
            [[len(cell) for cell in row] for row in item["A"]],
            [[len(cell) for cell in row] for row in item["B"]]) <= W_MAX
        fingerprints.add(fingerprint)
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct legal designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's optimized, invariant-distinct filling."""
    chosen = _design_bank(G)[idx]
    if chosen is None:
        raise ValueError("card found no invariant-distinct rate-legal filling")
    A, B = chosen
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: Component-only selection excludes mixed-sector minimum logicals; measured Z23 parent slot 20 retained a weight-14 X logical touching both sectors, including Arow1/Brow0 and Arow1/Brow1.
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
