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
CONCEPT = ("In cyclic Z29 of order 29, A=(2,4) and B=(2,4), partitioning distinct binomial gaps to reduce short component cycles should reduce circuit failures at k=132.")
GROUP = ("Keep the measured Z29 order-29 bed: its (2,4)x(2,4) fillings reached fitness 7.0367, so a filling-only comparison is informative.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: n=580, equal-length components, and guaranteed k>=116, rate>=0.200. All-binomial fillings measured k=132, rate 0.228 in cell 0.250. Parent failures involved varying sector-1 Acol/Bcol data blocks rather than one persistent check-block hotspot.")
FILLING = ("Keep sixteen distinct-element binomials per side and weight-12 rows. Assign forty non-automorphic gap spectra to low-short-relation row partitions, then screen independent labels for simple component four-, six-, and eight-cycles and verified k. Test for fewer stage-three fallthroughs and fewer than the parent's 4-14 Z failures per 200000 shots; the distance proxy is not fitness.")
FAMILY_NAME = "z29_24x24_partitioned_gaps_simple_cycles_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Additive cyclic group Z29.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 29 = 580


def _gap_canon(gaps):
    """Identify spectra related by a Z29 automorphism or gap reversal."""
    return min(
        tuple(sorted(min((u * d) % 29, 29 - (u * d) % 29)
                     for d in gaps))
        for u in range(1, 15)
    )


def _row_relations(gaps):
    """Count signed three- and four-gap zero sums modulo 29."""
    from itertools import combinations, product
    counts = []
    for size in (3, 4):
        hits = 0
        for word in combinations(gaps, size):
            for signs in product((-1, 1), repeat=size - 1):
                hits += (word[0] + sum(s * d for s, d
                                      in zip(signs, word[1:]))) % 29 == 0
        counts.append(hits)
    return tuple(counts)


def _make_gap_designs():
    """Choose different automorphism orbits and partition each into rows."""
    from itertools import combinations
    four_scores = {g: _row_relations(g)
                   for g in combinations(range(1, 15), 4)}
    designs, seen = [], set()
    for gaps in combinations(range(1, 15), 8):
        signature = _gap_canon(gaps)
        if signature in seen:
            continue
        seen.add(signature)
        best = None
        # Fix the first gap in row zero to avoid scoring both row orders.
        for chosen in combinations(gaps[1:], 3):
            first = (gaps[0],) + chosen
            first_set = set(first)
            second = tuple(d for d in gaps if d not in first_set)
            sa, sb = four_scores[first], four_scores[second]
            cost = (max(sa[0], sb[0]), sa[0] + sb[0],
                    max(sa[1], sb[1]), sa[1] + sb[1])
            if best is None or cost < best[0]:
                best = (cost, (first, second))
        global_cost = _row_relations(gaps)
        designs.append((best[0] + global_cost + (signature,),
                        best[1]))
    designs.sort(key=lambda item: item[0])
    assert len(designs) >= 40
    return tuple(partition for _, partition in designs[:40])


_GAP_DESIGNS = _make_gap_designs()


def _binomial_grid(rng, partition):
    """Randomize labels within a selected, distinct-gap row partition."""
    rows = list(partition)
    if rng.randrange(2):
        rows.reverse()
    grid = []
    for gaps in rows:
        row = []
        for gap in rng.sample(gaps, 4):
            start = rng.randrange(29)
            step = gap if rng.randrange(2) else -gap
            row.append([(start, 0), ((start + step) % 29, 0)])
        grid.append(row)
    return grid


def _component_defects(G, grid, side):
    """Short kernel relations and four-cycles of an actual lift action.

    A variable h meets check g*h; B uses h*g^-1. These are the
    component actions used by lifted_matrices, not cyclic-shift proxies.
    """
    columns = []
    for row_column in range(len(grid[0])):
        for h in G.elements:
            syndrome = 0
            for i, row in enumerate(grid):
                for g in row[row_column]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)

    pairs = {}
    twos = fours = cycles = 0
    for u, first in enumerate(columns):
        for second in columns[u + 1:]:
            syndrome = first ^ second
            twos += syndrome == 0
            fours += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            overlap = (first & second).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return twos, fours, cycles


def _component_cycles_68(G, grid, side):
    """Count simple six- and eight-cycles at both check-row orbit roots."""
    q = G.q
    check_vars = [[] for _ in range(2 * q)]
    var_checks = [[] for _ in range(4 * q)]
    for j in range(4):
        for h in G.elements:
            variable = j * q + G.idx(h)
            for i in range(2):
                for g in grid[i][j]:
                    check = G.mul(g, h) if side == "A" else G.mul(h, G.inv(g))
                    check = i * q + G.idx(check)
                    check_vars[check].append(variable)
                    var_checks[variable].append(check)

    sixes = eights = 0
    for root in (G.idx(G.e), q + G.idx(G.e)):
        root_vars = set(check_vars[root])
        paths = {}
        for first in check_vars[root]:
            for middle in var_checks[first]:
                if middle == root:
                    continue
                for second in check_vars[middle]:
                    if second == first:
                        continue
                    for end in var_checks[second]:
                        if end != root and end != middle:
                            paths.setdefault(end, []).append(
                                (first, second, middle))
        for end, entries in paths.items():
            closing = root_vars.intersection(check_vars[end])
            for p, (first, second, middle) in enumerate(entries):
                sixes += len(closing - {first, second})
                for other_first, other_second, other_middle in entries[p + 1:]:
                    if (middle != other_middle
                            and len({first, second, other_first,
                                     other_second}) == 4):
                        eights += 1
    return sixes, eights


def NEW_CTX(rng, G, idx):
    """Screen this slot's gap-orbit pair on both actual component actions."""
    profile = idx % 20
    tier = idx // 20
    gaps_a = _GAP_DESIGNS[profile]
    gaps_b = _GAP_DESIGNS[20 + profile]
    counts = [[2] * 4 for _ in range(2)]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    shortlist = []
    for _ in range(36 + 2 * tier):
        a = _binomial_grid(rng, gaps_a)
        b = _binomial_grid(rng, gaps_b)
        da = _component_defects(G, a, "A")
        db = _component_defects(G, b, "B")
        cost = (da[0] + db[0], max(da[2], db[2]),
                da[2] + db[2], max(da[1], db[1]),
                da[1] + db[1])
        shortlist.append((cost, rng.random(), a, b))
    shortlist.sort(key=lambda item: item[:2])

    best_key, best = None, None
    for cost, _, a, b in shortlist[:9]:
        sa, ea = _component_cycles_68(G, a, "A")
        sb, eb = _component_cycles_68(G, b, "B")
        k = lifted_k(G, a, b)
        # n=580; keep the measured mean in rate cell 0.250.
        key = (not 131 <= k <= 159, abs(k - 132),
               cost[:3], max(sa, sb), sa + sb,
               max(ea, eb), ea + eb, cost[3:])
        if best_key is None or key < best_key:
            best_key, best = key, {"A": a, "B": b}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return one complete, independently screened cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Uniform rolled D29 (1,3)x(1,3) did not protect the parent: its five
# measured slots had abundant weight-four logicals and at least one
# basis reached the 100-failure cap within 750 shots.
# Higher row weight does not always suppress X failures: the Z43
# parent's w12 slot 3 had two in 200000 shots; w11 slots 9 and 39 had zero.
# Component four-/six-cycle screening on Z31 (2,3)x(2,5) did not
# balance the bases: all five measured slots had zero X failures,
# while Z reached 100 failures in 66750-170250 shots.
# Short-relation screening on C9 semidirect C4 did not guarantee a
# zero-failure Z basis: its five measured slots had 2-8 Z failures
# per 200000 shots, and slots 23 and 40 also had an X failure.
# REFUTED: Distinct Z29 binomial gaps with the parent's short-relation
# and six-cycle-presence screen eliminate worse-basis failures: all five
# measured slots still had at least four Z failures in 200000 shots;
# slot 9 had fourteen.
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
