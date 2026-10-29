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
CONCEPT = ("Q8 x Z5, order 40, shapes (2,3)x(2,3): independently optimized noncommuting two-check components should eliminate the parent's weight-four logical families and improve the 0.100 cell's five-code fitness; exceeding its 54.9 incumbent is a measurement hypothesis.")
GROUP = ("Keep Q8 x Z5 and q=40 under L2. The parent produced five d30s=4 readings and only six distinct designs. The order-44 (2,3)x(2,3) beds reached fitness 54.9 and 48.4, motivating the shape rather than establishing a transferable group advantage. The unchanged group gives n=520.")
SHAPES = ("Target cell 0.100: A=B=(2,3), n=520, kb=40 and guaranteed rate 1/13. Both components have 120 variables and 80 checks. Parent minima were entirely in sector-1 Acol/Bcol blocks, especially Acol1 and Bcol0..2. Add a check row to each component and require verified 40<=k<=64, so every emitted rate is below 0.125.")
FILLING = ("Twenty recipes cross five degree profiles with four relative-order policies. Both CSS row sums are <=12. Avoid involution-separated terms and require noncommuting relatives in cells of size >=3. Extract components from verified lifts, exclude component kernels through weight four, refine actual kernel witnesses, then require full stabilizer-aware logical-four exclusion. Full-CSS witness probes select finalists; high readings remain upper bounds. Different combined-Tanner fingerprints exclude emitted orbit copies. Failed screens never substitute parent fillings.")
FAMILY_NAME = "r010_Q8xZ5_23x23_two_check_noncommuting_probe_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    # (a,b,c) = x^a y^b z^c:
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1; z central of order 5.
    if q != 40:
        raise ValueError("Q8 x Z5 requires order 40")
    els = [(a, b, c) for c in range(5)
           for b in range(2) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 2, g[2] % 5)

    def mul(g, h):
        a, b, c = g
        d, e, f = h
        return ((a + (-1 if b else 1)*d + 2*b*e) % 4,
                (b + e) % 2, (c + f) % 5)

    def inv(g):
        a, b, c = g
        return (((a + 2) if b else -a) % 4, b, (-c) % 5)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*4 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _q85_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


from collections import Counter
from functools import lru_cache


# Five intrinsic degree architectures, not row/column relabelings.
# Each pair satisfies both A-row+B-column and B-row+A-column <=12.
_Q85_DEGREES = (
    (((2, 2, 2), (2, 2, 2)),
     ((2, 2, 2), (2, 2, 2))),
    (((2, 2, 2), (2, 2, 2)),
     ((2, 2, 4), (2, 4, 2))),
    (((2, 2, 3), (2, 3, 2)),
     ((2, 2, 3), (2, 3, 2))),
    (((2, 2, 3), (2, 3, 2)),
     ((3, 3, 1), (2, 2, 3))),
    (((3, 3, 2), (1, 1, 4)),
     ((2, 2, 2), (2, 2, 2))),
)


def _q85_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _q85_components(hx, hz, q):
    """Extract both two-check components from the VERIFIED LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bm = (1 << (cb * q)) - 1
    br = [row & bm for row in hz[:rb * q]]
    return (_q85_columns(ar, ca * q),
            _q85_columns(br, cb * q))


def _q85_component_clean4(cols):
    """Exact component-kernel exclusion through weight four."""
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _q85_component_cycles(cols):
    total = 0
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            overlap = (ci & cj).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _q85_draw(rng, G, counts, mode, orders, pools):
    """Independent voltages with prescribed intrinsic relative orders."""
    grid = []
    for i, weights in enumerate(counts):
        row = []
        for j, weight in enumerate(weights):
            if weight == 1:
                row.append([G.pick(rng)])
                continue
            wanted = (20, 4, 5, 10)[mode] if (i + j) % 2 == 0 else 20
            # Two relatives cannot be noncommuting when the first is central.
            if weight == 3 and wanted in (5, 10):
                wanted = 20
            selected = None
            for _ in range(64):
                anchor = G.pick(rng)
                cell = [anchor, G.mul(anchor, rng.choice(pools[wanted]))]
                while len(cell) < weight:
                    legal = [
                        g for g in G.elements
                        if g not in cell and all(
                            orders[G.mul(G.inv(h), g)] > 2 for h in cell)
                    ]
                    if not legal:
                        break
                    cell.append(rng.choice(legal))
                if len(cell) != weight:
                    continue
                relatives = [G.mul(G.inv(anchor), g) for g in cell[1:]]
                if weight >= 3 and not any(
                    G.mul(g, h) != G.mul(h, g)
                    for t, g in enumerate(relatives) for h in relatives[:t]
                ):
                    continue
                selected = cell
                break
            if selected is None:
                raise ValueError("noncommuting cell recipe failed")
            row.append(selected)
        grid.append(row)
    return grid


def _q85_probe(cols, rng, trials, stabilizers=()):
    """Actual kernel/logical witnesses; a high reading is not a proof.

    Information-set elimination and XOR descent preserve the kernel.
    Full-CSS probes explicitly exclude the supplied stabilizer span.
    Stabilizer-row descent preserves each surviving logical class.
    """
    width = len(cols)
    basis = _q85_basis(stabilizers)
    best = width + 1
    order = list(range(width))

    def observe(word):
        nonlocal best
        if word and (not stabilizers or _q85_outside(word, basis)):
            best = min(best, word.bit_count())

    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
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
                observe(word)

        words = sorted(dependencies, key=int.bit_count)[:40]
        for i, word in enumerate(words):
            for other in words[:i]:
                observe(word ^ other)

        for start in words[:12]:
            if stabilizers and not _q85_outside(start, basis):
                continue
            word = start
            for _ in range(3):
                weight = word.bit_count()
                improved = word
                for other in words:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if candidate and cw < weight:
                        improved, weight = candidate, cw
                if improved == word:
                    break
                word = improved
            if stabilizers:
                if not _q85_outside(word, basis):
                    continue
                for _ in range(3):
                    changed = False
                    for row in stabilizers:
                        candidate = word ^ row
                        if candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
            observe(word)
        if best <= 4:
            break
    return best


def _q85_fingerprint(hx, hz, n):
    """Combined-Tanner invariant, even ignoring X/Z check colouring.

    Different fingerprints exclude incidence-graph isomorphism.
    Equal fingerprints are conservatively rejected, not declared isomorphic.
    """
    rows = hx + hz
    cols = _q85_columns(rows, n)
    adjacency = [Counter() for _ in rows]
    for syndrome in cols:
        ids = []
        while syndrome:
            bit = syndrome & -syndrome
            ids.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(ids):
            for b in ids[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for i, neighbors in enumerate(adjacency):
        square = {}
        for j, ij in neighbors.items():
            for t, jt in adjacency[j].items():
                square[t] = square.get(t, 0) + ij * jt
        triangle = sum(
            ij * square.get(j, 0) for j, ij in neighbors.items())
        profiles.append((
            rows[i].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(i, 0), triangle,
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _q85_two_check_bank(G):
    """Optimize twenty prescribed recipes and reject emitted orbit copies."""
    assert G.q == 40
    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("invalid group element order")
    pools = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (4, 5, 10, 20)
    }

    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        profile, mode = divmod(slot % 20, 4)
        sa, sb = _Q85_DEGREES[profile]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(10040023 + 100003 * slot)
        chosen = None

        for batch in range(2):
            finalists = []
            for attempt in range(40):
                A = _q85_draw(rng, G, sa, mode, orders, pools)
                B = _q85_draw(rng, G, sb, (mode + slot // 20) % 4,
                              orders, pools)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 520 or not 40 <= k <= 64:
                    continue
                ac, bc = _q85_components(hx, hz, G.q)
                if not _q85_component_clean4(ac):
                    continue
                if not _q85_component_clean4(bc):
                    continue
                da = _q85_probe(ac, rng, 6)
                db = _q85_probe(bc, rng, 6)
                cycles = (_q85_component_cycles(ac)
                          + _q85_component_cycles(bc))
                d = min(da, db)
                key = (k * d * d, d, da + db, -cycles)
                finalists.append({
                    "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db, "cycles": cycles,
                })
                finalists.sort(key=lambda item: item["key"], reverse=True)
                del finalists[5:]

            for item in finalists:
                item["da"] = min(
                    item["da"], _q85_probe(item["ac"], rng, 32))
                item["db"] = min(
                    item["db"], _q85_probe(item["bc"], rng, 32))
                d = min(item["da"], item["db"])
                item["key"] = (
                    item["k"] * d * d, d, -item["cycles"])
            finalists.sort(key=lambda item: item["key"], reverse=True)

            audited = []
            for item in finalists[:4]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                dx = _q85_probe(
                    _q85_columns(hz, n), rng, 6, stabilizers=hx)
                dz = _q85_probe(
                    _q85_columns(hx, n), rng, 6, stabilizers=hz)
                if dx > n or dz > n:
                    continue
                d = min(item["da"], item["db"], dx, dz)
                key = (item["k"] * d * d, d,
                       min(dx, dz), -cycles)
                audited.append((key, item))
            audited.sort(key=lambda pair: pair[0], reverse=True)

            for key, item in audited:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fingerprint = _q85_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= 12
                assert 0 < 8 * item["k"] < n
                fingerprints.add(fingerprint)
                chosen = (item["A"], item["B"], item["k"], key)
                break
            if chosen is not None:
                break
        # Failure voids this slot; it never emits an orbit or archived fallback.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the independently optimized, target-rate two-check recipe."""
    selected = _q85_two_check_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh low-rate design passed logical-four screening")
    A, B, k, key = selected
    return {"idx": idx, "card": idx % 20, "k": k,
            "selection": key, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly screened two-check filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Distinct-axis order-20 binomial differences plus bounded exact-four screening raised this Q8 x Z5 bed above distance four; all five measured current-parent codes still had d30s=4, including the two k=243 codes.
# REFUTED: Twenty common-C20 gap/unit cards supply twenty inequivalent designs: the measured parent produced only six distinct codes across sixty slots, with five d30s=4 leaders.
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
