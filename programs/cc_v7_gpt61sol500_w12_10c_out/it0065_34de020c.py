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
CONCEPT = ("Dihedral C23 semidirect C2, order 46, shapes (1,2)x(2,5): order-23 doubletons and verified component-kernel screening should produce top-five mean d30s>=10 and beat the 0.250-cell record of 17.3.")
GROUP = ("Replace Dic11 by C23 semidirect C2 with inversion action. Order 46 and order-two reflections make this genuinely different from the parent's order-44 dicyclic group. The related order-58 dihedral (1,2)x(2,4) bed scored 38.3; the new bed is untested, and that measurement does not establish its distance.")
SHAPES = ("Target cell 0.250 with (1,2)x(2,5): n=12*46=552, guaranteed k>=138 and rate>=1/4. Component checks are 46x92 and 92x230. Parent minima were entirely in sector-1 Acol/Bcol blocks, especially Acol1/2 and Bcol1/2. Replace both components and gate exact lifted k/n to [0.225,0.275).")
FILLING = ("Interleave support pairs distinguished by complete Tanner check/qubit degree multisets. Prioritize stronger B columns while checking both weight sums. Same-coset doubletons have relative order 23; larger cells explore balanced, uneven and uniform coset occupancy. Verified lifts supply ranks and component syndromes. Repeated kernel searches and exhaustive component words through weight six rescreen finalists; exact quantum weight<=4 exclusion is preferred with a legal fallback. Larger witness readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r025_D46_12x25_order23_degree_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # D46: x^23=y^2=1, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("C23 semidirect C2 requires order 46")
    els = [(a, b) for b in range(2) for a in range(23)]

    def red(g):
        return (g[0] % 23, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c)) % 23, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a) % 23, b)

    def idx(g):
        a, b = red(g)
        return 23*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _support_designs():
    """Degree-distinguished (1,2)x(2,5) supports, not labeling orbits."""
    from itertools import combinations_with_replacement

    aprofiles = ((4, 4), (3, 3), (3, 4))
    column_pairs = tuple(
        (a, b) for a in range(6) for b in range(6)
        if 2 <= a+b <= 6)
    buckets = {ta: {} for ta in aprofiles}

    for pairs in combinations_with_replacement(column_pairs, 5):
        # Columns are already canonical; also quotient by B-row exchange.
        swapped = tuple(sorted((b, a) for a, b in pairs))
        if pairs > swapped:
            continue
        br = (sum(a for a, b in pairs), sum(b for a, b in pairs))
        if min(br) < 7 or max(br) > 9:
            continue
        bc = tuple(a+b for a, b in pairs)
        tb = tuple(a for a, b in pairs) + tuple(b for a, b in pairs)

        for ta in aprofiles:
            ar = sum(ta)
            # HX: A-row plus B-column. HZ: B-row plus A-column.
            if ar + max(bc) > 12 or max(br) + max(ta) > 12:
                continue

            checks = tuple(sorted(
                [ar+c for c in bc] +
                [r+a for r in br for a in ta]))
            qubits = tuple(sorted(
                [a+c for a in ta for c in bc] +
                [ar+r for r in br]))
            fingerprint = (checks, qubits)
            priority = (
                min(bc), min(br), sum(tb),
                -sum(abs(a-b) for a, b in pairs),
                -max(tb), -tb.count(0))
            item = (priority, fingerprint, ta, tb)
            previous = buckets[ta].get(fingerprint)
            if previous is None or item > previous:
                buckets[ta][fingerprint] = item

    lists = {
        ta: sorted(bucket.values(), reverse=True)
        for ta, bucket in buckets.items()}
    positions = {ta: 0 for ta in aprofiles}
    designs, seen = [], set()
    while True:
        added = False
        for ta in aprofiles:
            items = lists[ta]
            while positions[ta] < len(items):
                _, fingerprint, aa, bb = items[positions[ta]]
                positions[ta] += 1
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                designs.append((aa, bb))
                added = True
                break
        if not added:
            break

    # Each degree occurs q times in the lift. Unequal fingerprints
    # therefore exclude Tanner-graph isomorphism between these cards.
    if len(designs) < 20:
        raise ValueError("fewer than twenty degree-distinguished designs")
    return tuple(designs)


_SUPPORT_DESIGNS = _support_designs()


def _four_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for a, x in enumerate(bits):
            for y in bits[a+1:]:
                key = (x, y)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_probe(rows, width, rng):
    """Upper readings on classical kernel distance, used only as heuristics."""
    best = width + 1
    for _ in range(4):
        perm = list(range(width))
        rng.shuffle(perm)
        pivots = {}
        for original in rows:
            row = 0
            while original:
                bit = original & -original
                row ^= 1 << perm[bit.bit_length()-1]
                original ^= bit
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for f in range(width):
            if f in pivots:
                continue
            v = 1 << f
            for p, row in ordered:
                if (row & v).bit_count() & 1:
                    v ^= 1 << p
            best = min(best, v.bit_count())
            basis.append(v)
        for a, v in enumerate(basis):
            for u in basis[a+1:]:
                best = min(best, (v ^ u).bit_count())
    return best


def _component_checks(hx, hz, q):
    """Extract both component actions only from the VERIFIED LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    arows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i*cb)*q+r]
            packed = 0
            for j in range(ca):
                packed |= ((source >> (j*cb*q)) & mask) << (j*q)
            arows.append(packed)
    bmask = (1 << (cb*q))-1
    brows = [hz[a*q+r] & bmask for a in range(rb) for r in range(q)]
    return arows, brows


def _label_grid(rng, G, counts, shape, model):
    """Long-order doubletons; larger cells explore three occupancy models."""
    rows, cols = shape
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            count = counts[cols*i+j]
            phase = rng.randrange(2)
            if count <= 2:
                bs = [phase]*count
            elif model == 0:
                bs = [phase]*((count+1)//2) + [1-phase]*(count//2)
            elif model == 1:
                bs = [phase]*(count-1) + [1-phase]
            else:
                bs = [rng.randrange(2) for _ in range(count)]
            # Disjoint cosets and sampling without replacement ensure
            # distinct elements, including zero cells and repeated cosets.
            cell = [
                (a, b) for b in range(2)
                for a in rng.sample(range(23), bs.count(b))]
            row.append(cell)
        grid.append(row)
    return grid


def _syndrome_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _kernel_reading(cols, rng, trials):
    """Actual classical kernel witnesses; never a distance certificate."""
    width = len(cols)
    best = width+1
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        # Every operation preserves the true component kernel.
        for sweep in range(2):
            rng.shuffle(words)
            changed = False
            for a in range(len(words)):
                v, weight = words[a], words[a].bit_count()
                for b, u in enumerate(words):
                    if a == b:
                        continue
                    z = v ^ u
                    zw = z.bit_count()
                    if zw:
                        best = min(best, zw)
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                words[a] = v
                best = min(best, weight)
            if not changed:
                break
        if best <= 6:
            break
    return best


def _short_component_reading(cols):
    """Find a kernel witness through weight six, or exhaustively return 7.

    A returned witness weight is an upper reading, not necessarily the
    minimum. Sentinel 7 means no nonzero kernel word of weight <=6.
    This tests the classical kernel, not the quantum logical quotient.
    """
    if 0 in cols:
        return 1
    singles = set(cols)
    if len(singles) != len(cols):
        return 2

    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles:
                return 3
            if s in pairs:
                return 4
            pairs.add(s)

    triples = set()
    for i, x in enumerate(cols):
        for j in range(i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if s in pairs:
                    return 5
                if s in triples:
                    return 6
                triples.add(s)
    return 7


def _has_small_logical(checks, stabilizers, width):
    """Exact existence test for a non-stabilizer logical of weight 1..4."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length()-1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    cols = _syndrome_columns(checks, width)
    seen = {0: 0}
    bits = [1 << i for i in range(width)]
    for i, s in enumerate(cols):
        old = seen.get(s)
        if old is None:
            seen[s] = bits[i]
        elif outside(bits[i] ^ old):
            return True

    for i, si in enumerate(cols):
        for j in range(i):
            s = si ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif outside(word ^ old):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Optimize a degree-distinguished support card for rate cell 0.250."""
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta)]
    sb = [list(tb[:5]), list(tb[5:])]
    assert G.q == 46
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    kb = 3*G.q

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    finalists, seen = [], set()
    for attempt in range(96):
        # All three occupancy rules are constructive alternatives,
        # not automorphism images of a fixed labeling.
        model = (idx + attempt) % 3
        A = _label_grid(rng, G, ta, A_SHAPE, model)
        B = _label_grid(rng, G, tb, B_SHAPE, model)
        key = tuple(tuple(sorted(cell))
                    for grid in (A, B) for row in grid for cell in row)
        if key in seen:
            continue
        seen.add(key)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Exact target gate; the lower bound itself is rate 0.250.
        if not (k >= kb and 9*n <= 40*k < 11*n):
            continue
        ar, br = _component_checks(hx, hz, G.q)
        ac = _syndrome_columns(ar, 2*G.q)
        bc = _syndrome_columns(br, 5*G.q)
        item = {
            "A": A, "B": B, "k": k,
            "hx": hx, "hz": hz, "n": n,
            "ac": ac, "bc": bc,
            "da": _kernel_reading(ac, rng, 2),
            "db": _kernel_reading(bc, rng, 2),
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[6:]
        if attempt >= 47 and len(finalists) == 6:
            break

    if not finalists:
        raise ValueError("no positive-k labeling in target rate cell")

    # Independent deeper searches reduce selection of lucky readings.
    for item in finalists:
        item["da"] = min(item["da"],
                         _kernel_reading(item["ac"], rng, 12))
        item["db"] = min(item["db"],
                         _kernel_reading(item["bc"], rng, 12))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:3]

    for item in finalists:
        item["da"] = min(item["da"],
                         _kernel_reading(item["ac"], rng, 48))
        item["db"] = min(item["db"],
                         _kernel_reading(item["bc"], rng, 48))

        # Screen both verified component kernels through weight six.
        # A detected classical word need not be a quantum logical.
        da = _short_component_reading(item["ac"])
        db = _short_component_reading(item["bc"])
        if da <= 6:
            item["da"] = min(item["da"], da)
        if db <= 6:
            item["db"] = min(item["db"], db)

    finalists.sort(key=merit, reverse=True)
    winner = finalists[0]
    for item in finalists:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        if _has_small_logical(hx, hz, n):
            continue
        if _has_small_logical(hz, hx, n):
            continue
        winner = item
        break

    # Keep a legal code even if no finalist passes the optional
    # quantum screen; that event refutes the filling hypothesis,
    # rather than silently repairing or wasting the slot.
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert n == 552 and k >= kb and 9*n <= 40*k < 11*n
    assert max(row.bit_count() for row in hx+hz) <= 12
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell without further label changes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
