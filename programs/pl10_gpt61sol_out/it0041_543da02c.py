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
CONCEPT = ("On metacyclic Z7:Z2(s=6), order 14, shapes (1,4)x(2,7), odd-weight A columns and triangle-free degree-two B components should improve worse-basis circuit reliability by replacing ubiquitous weight-three vulnerabilities with balanced weight-four protection.")
GROUP = ("Keep the frozen order-14 metacyclic group and multiplication rule. The related (2,4)x(2,5) bed scored 4.7, whereas the current bed scores 3.2219; this motivates changing its element model without claiming a group-level limitation.")
SHAPES = ("Keep (1,4)x(2,7) for cell 0.500: lift order 14 gives n=420 and guaranteed k>=210, with unequal component matrices 14x56 and 28x98. Current witnesses are sector-1 Acol/Bcol operators, and failure enrichment is diffuse at 1.0-1.2x across A columns and B columns; no stable Acol/Brow hotspot justifies local reinforcement.")
FILLING = ("A counts become [1,3,3,3]; B columns each contain two terms, with row sums 7/7 or 8/6. Twenty briefs cross four pure/mixed A-trinomial profiles with five B graph designs. Verified lifts reject duplicate columns and component triangles; component weight-four multiplicity and concentration shortlist candidates, then exact nontrivial CSS weight-four counts and actual ranks select them. Later batches change minimax, total, or X-weighted priorities. HX weight is 12, HZ weight <=11, and extraction uses 13440 CNOTs. The intended improvement is fewer circuit failures than the parent's 108-139 X failures per 250 shots, not distance alone.")
FAMILY_NAME = "r050_m7r2s6_14x27_oddA_trianglefreeB_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def NEW_CTX(rng, G, idx):
    """Twenty structural briefs; screen component and actual CSS kernels."""
    pure, bcard = (idx % 20) // 5, idx % 5
    priority = (idx // 20) % 3
    q, n = G.q, 30 * G.q
    identity = G.e

    def columns(rows):
        result = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def component_four(cs):
        """Exact classical weight-four count and worst variable load.
        Distinct nonzero columns make equal-syndrome pairs disjoint.
        Each weight-four support has exactly three pair partitions."""
        if 0 in cs or len(set(cs)) != len(cs):
            return None
        singles = set(cs)
        counts, pairs = {}, []
        for a, x in enumerate(cs):
            for b in range(a):
                syndrome = x ^ cs[b]
                if syndrome in singles:
                    return None
                counts[syndrome] = counts.get(syndrome, 0) + 1
                pairs.append((a, b, syndrome))
        collisions = sum(c * (c - 1) // 2 for c in counts.values())
        load = [0] * len(cs)
        for a, b, syndrome in pairs:
            c = counts[syndrome] - 1
            load[a] += c
            load[b] += c
        assert collisions % 3 == 0
        return collisions // 3, max(load) // 3

    def css_four(checks, stabilizers):
        """Exact nontrivial CSS weight-four count, rejecting weights 1..3."""
        cs = columns(checks)
        if 0 in cs or len(set(cs)) != n:
            return None
        singles = set(cs)

        # A canonical quotient map modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        residue = [0] * n
        for a in range(n):
            row = pivots.get(a)
            if row is None:
                residue[a] = 1 << a
            else:
                tail, value = row ^ (1 << a), 0
                while tail:
                    bit = tail & -tail
                    value ^= residue[bit.bit_length() - 1]
                    tail ^= bit
                residue[a] = value

        totals, frequencies = {}, {}
        logical_partitions = 0
        for a, x in enumerate(cs):
            for b in range(a):
                syndrome = x ^ cs[b]
                if syndrome in singles:
                    return None
                value = residue[a] ^ residue[b]
                key = (syndrome, value)
                total = totals.get(syndrome, 0)
                same = frequencies.get(key, 0)
                # Equal syndrome but unequal quotient residue is logical.
                logical_partitions += total - same
                totals[syndrome] = total + 1
                frequencies[key] = same + 1
        assert logical_partitions % 3 == 0
        return logical_partitions // 3

    def make_A():
        row = [[identity]]
        for j in range(3):
            if j < pure:
                a, b = rng.sample(range(1, 7), 2)
                row.append([identity, (a, 0), (b, 0)])
            else:
                row.append([identity, (rng.randrange(1, 7), 0),
                            (rng.randrange(7), 1)])
        return [row]

    def independent_coset(step, parity):
        # Three mutually nonadjacent vertices of a seven-cycle.
        shift = rng.randrange(7)
        return [((shift + t * step) % 7, parity) for t in (0, 2, 4)]

    def make_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        if bcard < 3:
            # Three genuinely different bi-Cayley connection profiles.
            nr = (3, 2, 1)[bcard]
            ratios = (rng.sample([(a, 0) for a in range(7)], 7 - nr) +
                      rng.sample([(a, 1) for a in range(7)], nr))
            start = 0
        else:
            step = rng.choice((1, 2, 3))
            B[0][0] = [identity, (step, 0)]
            rotations = independent_coset(step, 0)
            reflections = independent_coset(step, 1)
            if bcard == 3:
                # One internal cycle: component check degrees 8 and 6.
                ratios = rotations + reflections
                start = 1
            else:
                # Two internal cycles, one on each check-row orbit.
                # Matching steps permit a triangle-free cross connection.
                B[1][1] = [identity, (step, 0)]
                ratios = rotations + rng.sample(reflections, 2)
                start = 2
        rng.shuffle(ratios)
        for j, ratio in enumerate(ratios, start):
            B[0][j] = [identity]
            B[1][j] = [ratio]
        return B

    def objective(x, z, concentration):
        if priority == 0:
            return (max(x, z), x + z, concentration)
        if priority == 1:
            return (x + z, max(x, z), concentration)
        return (max(2 * x, z), 2 * x + z, concentration)

    shortlist = []
    signatures = set()
    for trial in range(160):
        A, B = make_A(), make_B()
        hx, hz, actual_n = lifted_matrices(G, A, B)
        assert actual_n == n
        cx, cz = columns(hx), columns(hz)

        # Component incidence columns extracted only from verified lifts.
        ac = [cx[(j * 7) * q + r] for j in range(4) for r in range(q)]
        bc = cz[:7 * q]
        if any(c.bit_count() != 2 for c in bc):
            continue
        astat, bstat = component_four(ac), component_four(bc)
        if astat is None or bstat is None:
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)

        signature = (tuple(sorted(hx)), tuple(sorted(hz)))
        if signature in signatures:
            continue
        signatures.add(signature)
        a4, aload = astat
        b4, bload = bstat
        # Block-repetition estimates only shortlist; actual CSS counts
        # below determine the final choice.
        key = objective(4 * b4, 7 * a4,
                        max(4 * bload, 7 * aload))
        shortlist.append((key, trial, A, B, hx, hz))
        shortlist.sort(key=lambda item: (item[0], item[1]))
        del shortlist[8:]

    best, best_key = None, None
    for component_key, trial, A, B, hx, hz in shortlist:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.475 <= k / n < 0.525):
            continue
        x4 = css_four(hz, hx)
        if x4 is None:
            continue
        z4 = css_four(hx, hz)
        if z4 is None:
            continue
        key = objective(x4, z4, component_key[2]) + (k - 210,)
        if best_key is None or key < best_key:
            best_key = key
            best = (A, B, hx, hz)
    if best is None:
        raise ValueError("no triangle-free, short-logical-screened filling")
    assert css_ok(best[2], best[3])
    return {"A": best[0], "B": best[1],
            "brief": (pure, bcard, priority), "screen": best_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell; no orbit-copy randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational A-binomial steps plus raw syndrome-ambiguity screening did not eliminate three-body logical vulnerabilities on (1,4)x(2,7): measured slots retained hundreds of weight-three sector-1 witnesses, 108-139 X failures per 250 shots, and fitness 3.2219.
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
