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
CONCEPT = ("On dihedral Z29:Z2(s=28), order 58, A=(1,3) and B=(1,3), replacing one mixed trinomial per side with an asymmetric rotation-only trinomial should produce five novel distance readings at least eight and test escape from the observed all-mixed weight-eight plateau.")
GROUP = ("Frozen byte-for-byte under L1. This bed improved from the rolled seed's fitness 6.4 to 25.6 through filling changes, providing measured evidence that its element model matters; no group change is needed for this experiment.")
SHAPES = ("Frozen (1,3)x(1,3): n=580, guaranteed k>=232 and rate>=0.400, with equal length-174/58-check components. The seed's measured minima occupied sector-1 Acol/Bcol blocks. Screen both components and mixed-sector supports, and require verified k=232.")
FILLING = ("Every cell has three distinct terms, giving check weight 12. Each side contains one non-arithmetic-progression rotation triple and two mixed triples. Slots occupy disjoint full-Tanner invariant buckets, not orbit copies. Within each bucket prefer verified fillings without kernels through weight six; retain a legal bucket-specific fallback if stronger screening fails. Larger evaluator readings remain upper bounds.")
FAMILY_NAME = "r040f_m29r2s28_13x13_puremix_bucket_kernel6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # metacyclic Z_29 x| Z_2: y x y^-1 = x^28 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
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

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize a novel pure/mixed architecture in an invariant bucket."""
    from collections import Counter
    from hashlib import blake2b

    q = G.q
    n1 = 9 * q
    mask = (1 << q) - 1
    # Representatives of the four non-AP affine triple classes over F29.
    geometries = (3, 4, 5, 9)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                out[low.bit_length() - 1] |= bit
                row ^= low
        return out

    def free_four(cols):
        # Every column here has odd weight, excluding odd kernels.
        if 0 in cols or len(set(cols)) != len(cols):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                overlap = (x & cols[j]).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def bucket(hx, hz, xc, zc, n):
        """Uncolored union-Tanner invariant, including local refinements.

        Isomorphic graphs have equal fingerprints and therefore equal
        buckets. Different emitted slot buckets cannot be isomorphic.
        """
        rows = hx + hz
        uc = [x | (z << len(hx)) for x, z in zip(xc, zc)]
        pair_counts = {}
        incidences = []
        row_keys = []
        for row in rows:
            bits = []
            while row:
                low = row & -row
                bits.append(low.bit_length() - 1)
                row ^= low
            incidences.append(bits)
            keys = []
            for a, x in enumerate(bits):
                for y in bits[a + 1:]:
                    key = x * n + y
                    keys.append(key)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
            row_keys.append(keys)

        rc = [sum(pair_counts[key] - 1 for key in keys)
              for keys in row_keys]
        qc = [0] * n
        for key, count in pair_counts.items():
            if count > 1:
                x, y = divmod(key, n)
                value = count * (count - 1) // 2
                qc[x] += value
                qc[y] += value

        qlabels = [(uc[c].bit_count(), qc[c]) for c in range(n)]
        rlabels = [
            (len(bits), rc[r],
             tuple(sorted(Counter(qlabels[c] for c in bits).items())))
            for r, bits in enumerate(incidences)
        ]
        qrefined = []
        for c, col in enumerate(uc):
            neighbours = []
            while col:
                low = col & -col
                r = low.bit_length() - 1
                neighbours.append((len(incidences[r]), rc[r]))
                col ^= low
            qrefined.append(
                (qlabels[c], tuple(sorted(Counter(neighbours).items()))))
        fingerprint = (
            tuple(sorted(Counter(rlabels).items())),
            tuple(sorted(Counter(qrefined).items())),
        )
        digest = blake2b(repr(fingerprint).encode(),
                         digest_size=8).digest()
        return int.from_bytes(digest, "little") % N_SLOTS

    def has_six(cols):
        """Exact component weight-six existence after free_four.

        The verified component has a transitive action on each fibre.
        Translate a six-support to contain 0, q or 2q, then match
        disjoint triples. Only the component extraction uses LP conventions.
        """
        length = len(cols)
        bits = [1 << c for c in range(length)]
        anchored = {}
        for a in (0, q, 2 * q):
            for b in range(length):
                if b == a:
                    continue
                ab = cols[a] ^ cols[b]
                support = bits[a] | bits[b]
                for c in range(b + 1, length):
                    if c == a:
                        continue
                    anchored.setdefault(ab ^ cols[c], []).append(
                        support | bits[c])
        lookup = anchored.get
        for a in range(length):
            for b in range(a):
                ab = cols[a] ^ cols[b]
                support = bits[a] | bits[b]
                for c in range(b):
                    mates = lookup(ab ^ cols[c])
                    if mates:
                        triple = support | bits[c]
                        if any(not (triple & other) for other in mates):
                            return True
        return False

    def mixed_six(cols, x_checks):
        """Exact mixed-sector six-support test when all cells are units.

        An odd number of tail qubits requires odd sector-1 occupancy
        in each of the three check blocks. At total weight six this
        leaves only tail/sector-1 splits 1+5 and 3+3.
        An even nonzero tail support cannot vanish in an empty check
        block because each tail block map is invertible.
        """
        singles = [{} for _ in range(3)]
        for c in range(n1):
            block = (c // q) % 3 if x_checks else (c // q) // 3
            syndrome = (cols[c] >> (block * q)) & mask
            singles[block][syndrome] = 1 << c

        pair_maps = [None] * 3
        tail = [
            tuple((s >> (b * q)) & mask for b in range(3))
            for s in cols[n1:]
        ]

        # One tail qubit, with three sector-1 qubits in one block
        # and one in each of the other two.
        for segments in tail:
            for heavy in range(3):
                others = [b for b in range(3) if b != heavy]
                if any(segments[b] not in singles[b] for b in others):
                    continue
                if pair_maps[heavy] is None:
                    items = list(singles[heavy].items())
                    pairs = {}
                    for a, (s, v) in enumerate(items):
                        for t, w in items[:a]:
                            pairs[s ^ t] = v | w
                    pair_maps[heavy] = pairs
                pairs = pair_maps[heavy]
                for syndrome, support in singles[heavy].items():
                    mate = pairs.get(segments[heavy] ^ syndrome)
                    if mate is not None and not (mate & support):
                        return True

        # Three tail qubits and one sector-1 qubit in each block.
        for a in range(q):
            for b in range(a):
                ab = tuple(tail[a][h] ^ tail[b][h] for h in range(3))
                for c in range(b):
                    if all((ab[h] ^ tail[c][h]) in singles[h]
                           for h in range(3)):
                        return True
        return False

    pool = []
    admitted = 0
    attempts = max(256, 20 * N_SLOTS)
    for attempt in range(attempts):
        # Change triangle classes as well as gaps during optimization.
        # The invariant bucket, not random offsets, defines slot identity.
        ta = geometries[(idx + attempt // 48) % 4]
        tb = geometries[(idx // 4 + attempt // 24) % 4]
        scale_b = rng.randrange(1, 29)
        gaps = rng.sample(range(1, 15), 4)
        cells = {}
        for side, triangle, scale, ds in (
                ("A", ta, 1, gaps[:2]),
                ("B", tb, scale_b, gaps[2:])):
            row = [[(0, 0), (scale, 0),
                    ((scale * triangle) % 29, 0)]]
            for gap in ds:
                row.append([(0, 0), (gap, 0),
                            (rng.randrange(29), 1)])
            cells[side] = [row]

        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        xc, zc = columns(hx, n), columns(hz, n)
        ac = [xc[3 * j * q + h]
              for j in range(3) for h in range(q)]
        bc = zc[:3 * q]
        if not free_four(ac) or not free_four(bc):
            continue
        if bucket(hx, hz, xc, zc, n) != idx:
            continue
        if n - gf2_rank(hx) - gf2_rank(hz) != 4 * q:
            continue

        ca, cb = cycles(ac), cycles(bc)
        pool.append((max(ca, cb), ca + cb, attempt, cells,
                     hx, hz, xc, zc, ac, bc))
        admitted += 1
        if admitted >= 4:
            break

    if not pool:
        raise ValueError("no exact-rate filling in prescribed Tanner bucket")
    pool.sort(key=lambda item: item[:3])

    best = None
    best_key = None
    for ca, total, attempt, cells, hx, hz, xc, zc, ac, bc in pool:
        if not free_four(xc) or not free_four(zc):
            continue

        # Verify invertibility of every individual cell map by extracting
        # its columns from the scaffold's component matrices.
        unit_cells = all(
            gf2_rank(columns(component[j*q:(j+1)*q], q)) == q
            for component in (ac, bc) for j in range(3))
        a6, b6 = has_six(ac), has_six(bc)
        if unit_cells:
            z6 = mixed_six(xc, True)
            x6 = mixed_six(zc, False)
        else:
            # Do not claim exhaustive mixed-sector screening without units.
            x6 = z6 = True
        key = (int(a6) + int(b6) + int(x6) + int(z6),
               ca, total, attempt)
        if best_key is None or key < best_key:
            best_key, best = key, (cells, hx, hz, unit_cells)
        if key[0] == 0:
            break

    if best is None:
        # Legal, distinct-bucket fallback; distance is measured downstream.
        _, _, _, cells, hx, hz, _, _, _, _ = pool[0]
        screened = False
    else:
        cells, hx, hz, units = best
        screened = units and best_key[0] == 0
    assert css_ok(hx, hz)
    assert 580 - gf2_rank(hx) - gf2_rank(hz) == 232
    return {"cells": cells, "screened_through_six": screened,
            "selection": best_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without additional sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled counts with uniform labels reliably suppress short logicals on this bed; many seed slots returned d2s=2 and all five measured seed codes returned d30s=4, with sector-1 minima.
# REFUTED: The shown all-mixed trinomial gap screens reliably exceed measured distance eight; all three shown best programs scored 25.6 with five d30s=8 readings, motivating a different pure/mixed architecture.
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
