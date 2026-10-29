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
CONCEPT = ("In cyclic Z36 of order 36 with A=(2,4), B=(2,3), independently screened binomial gap spectra should yield five distinct rate-0.150 codes with fewer worst-basis circuit failures.")
GROUP = ("Keep cyclic Z36 and its order-36 lift: the measured parent bed scored 7.2168 and included a zero-failure slot. The new shapes still give n=576, isolating the effect of exchanging the unequal component lengths.")
SHAPES = ("Use (2,4)x(2,3): n=576, guaranteed k>=72 and rate>=0.125; the parent's measured k=86-90 lies in cell 0.150. A has 144 component variables and B has 108, reversing the parent's unequal lengths; equal shapes would have equal-length components. Parent Z failures enriched differing sector-1 Acol/Bcol blocks, including Acol0/Bcol2 and Acol2/Bcol3, without a persistent hotspot.")
FILLING = ("Two distinct shifts per cell give HX/HZ row weights 12/10, reversing the parent's 10/12 balance. Twenty indexed A-gap orbits paired with B spectra, across three parity tiers, receive independent labels. Screen verified HX short traps, component relations and target-cell k to test whether Z failures and decoder fall-throughs decrease; distance proxy is only a selection gate.")
FAMILY_NAME = "r015_z36_24x23_exchanged_spectral_traps_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # Additive cyclic group Z36, encoded as one-coordinate tuples.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb); n = 16 * 36 = 576


def _component_short_cycles(G, cells, side):
    """Count rooted simple Tanner cycles of lengths four, six and eight.

    Group translation maps every lift coordinate of a given check row
    onto every other one, so one root per protograph check row suffices.
    Use the component's actual left/right lift conventions.
    """
    q = G.q
    check_vars = [[] for _ in range(2 * q)]
    var_checks = [[] for _ in range(len(cells[0]) * q)]
    for i, row in enumerate(cells):
        for r in G.elements:
            check = i * q + G.idx(r)
            for j, cell in enumerate(row):
                for g in cell:
                    c = G.mul(G.inv(g), r) if side == "A" else G.mul(r, g)
                    variable = j * q + G.idx(c)
                    check_vars[check].append(variable)
                    var_checks[variable].append(check)

    adjacent = [[] for _ in check_vars]
    for check, variables in enumerate(check_vars):
        for variable in variables:
            adjacent[check].extend(
                (other, variable) for other in var_checks[variable]
                if other != check
            )

    fours = sixes = eights = 0
    identity = G.idx(G.e)
    for i in range(2):
        root = i * q + identity
        root_edges = {}
        for other, variable in adjacent[root]:
            old = root_edges.setdefault(other, [])
            fours += len(old)
            old.append(variable)

        paths = {}
        for middle, first_var in adjacent[root]:
            for end, second_var in adjacent[middle]:
                if end == root or second_var == first_var:
                    continue
                for last_var in root_edges.get(end, ()):
                    if last_var != first_var and last_var != second_var:
                        sixes += 1

                old_paths = paths.setdefault(end, [])
                for old_middle, old_first, old_second in old_paths:
                    if old_middle != middle and len({
                        first_var, second_var, old_first, old_second
                    }) == 4:
                        eights += 1
                old_paths.append((middle, first_var, second_var))
    return fours, sixes, eights


def _gap_spectra36(n_odd, n_even):
    """Represent distinct undirected-gap orbits under Z36 units."""
    from itertools import combinations
    odd = (1, 5, 7, 11, 13, 17)
    even = (2, 4, 8, 10, 14, 16)
    units = (1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35)
    seen, spectra = set(), []
    for o in combinations(odd, n_odd):
        for e in combinations(even, n_even):
            gaps = tuple(sorted(o + e))
            signature = min(
                tuple(sorted(min((u * d) % 36, (-u * d) % 36)
                             for d in gaps))
                for u in units
            )
            if signature not in seen:
                seen.add(signature)
                spectra.append(signature)
    assert len(spectra) >= 20
    return tuple(sorted(spectra))


_A_SPECTRA36 = _gap_spectra36(4, 4)  # eight cells in A
_B_SPECTRA36 = _gap_spectra36(3, 3)  # six cells in B


def _trial_cells(rng, G, idx):
    """Index twenty A orbits, then apply one of three parity-tier motifs."""
    odd = (1, 5, 7, 11, 13, 17)
    even = (2, 4, 8, 10, 14, 16)
    design, tier = idx % 20, (idx // 20) % 3
    a_gaps = list(_A_SPECTRA36[design])
    b_gaps = list(_B_SPECTRA36[(7 * design + 11) %
                               len(_B_SPECTRA36)])

    def raise_odd_count(gaps):
        for _ in range(tier):
            available = [d for d in odd if d not in gaps]
            removable = [d for d in gaps if d in even]
            old = rng.choice(removable)
            new = rng.choice(available)
            gaps.remove(old)
            gaps.append(new)

    raise_odd_count(a_gaps)  # four, five or six odd gaps of eight
    raise_odd_count(b_gaps)  # three, four or five odd gaps of six

    def fill(gaps, width):
        rng.shuffle(gaps)
        grid = []
        for i in range(2):
            row = []
            for j in range(width):
                start = rng.randrange(G.q)
                gap = gaps[i * width + j]
                if rng.randrange(2):
                    gap = -gap
                row.append([(start, 0), ((start + gap) % G.q, 0)])
            grid.append(row)
        return grid

    return fill(a_gaps, 4), fill(b_gaps, 3)


def _component_pair_collisions(G, cells, side):
    """Count repeated two-column syndromes under the actual group action."""
    columns = []
    for j in range(len(cells[0])):
        for h in G.elements:
            syndrome = 0
            for i, row in enumerate(cells):
                for g in row[j]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)
    seen = {}
    collisions = 0
    for i, left in enumerate(columns):
        for right in columns[:i]:
            value = left ^ right
            collisions += seen.get(value, 0)
            seen[value] = seen.get(value, 0) + 1
    return collisions


def _verified_check_traps(rows, q):
    """Rooted four- and simple six-cycles of the actual lifted checks."""
    fours, sixes = [], []
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        nearby = []
        four = 0
        for index, row in enumerate(rows):
            if index == root_index:
                continue
            shared = root & row
            if shared:
                count = shared.bit_count()
                four += count * (count - 1) // 2
                nearby.append((index, shared, count))
        six = 0
        for a, (u, ru, nu) in enumerate(nearby):
            for v, rv, nv in nearby[:a]:
                outside = rows[u] & rows[v] & ~root
                if outside:
                    six += (nu * nv - (ru & rv).bit_count()) * outside.bit_count()
        fours.append(four)
        sixes.append(six)
    return max(fours), sum(fours), max(sixes), sum(sixes)


def NEW_CTX(rng, G, idx):
    """Screen a complete, indexed design using the verified CSS matrices."""
    sa = [[2] * 4 for _ in range(2)]
    sb = [[2] * 3 for _ in range(2)]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    finalists = []
    for _ in range(24 + idx % 7):
        A, B = _trial_cells(rng, G, idx)
        ac = _component_short_cycles(G, A, "A")
        bc = _component_short_cycles(G, B, "B")
        cycle_key = (
            max(ac[0], bc[0]), ac[0] + bc[0],
            max(ac[1], bc[1]), ac[1] + bc[1],
            max(ac[2], bc[2]), ac[2] + bc[2],
            rng.random()
        )
        finalists.append((cycle_key, A, B))
    finalists.sort(key=lambda entry: entry[0])

    chosen = None
    for cycle_key, A, B in finalists[:8]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        x_traps = _verified_check_traps(hx, G.q)
        z_traps = _verified_check_traps(hz, G.q)
        defects = (_component_pair_collisions(G, A, "A"),
                   _component_pair_collisions(G, B, "B"))
        # At n=576, k=82..100 stays within the requested rate cell.
        # HX detects the Z errors that limited the parent.
        key = (not 82 <= k <= 100,
               x_traps[2], x_traps[0], x_traps[3], x_traps[1],
               max(defects), sum(defects),
               z_traps[2], z_traps[0],
               cycle_key[:4], -k, cycle_key[4:])
        if chosen is None or key < chosen[0]:
            chosen = (key, A, B)
    return {"A": chosen[1], "B": chosen[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: cyclic-order-21 rolled sparse fillings do not consistently
# suppress sector-1 weight-six logicals or worst-basis failures: the five
# measured parent slots scored only 3.91-4.63.
# Refuted: minimizing component four-cycles alone consistently prevents
# circuit failures on this D23 bed. Parent slot 42 had six X-basis failures
# in 200000 shots despite w10 and a d2s estimate of 22.
# Refuted: balanced-cycle D23 binomials guarantee zero failures; current
# slots 2, 6 and 27 had respectively 1, 2 and 1 Z-basis failed shots.
# Refuted: Z37 component-cycle screening guarantees zero Z failures:
# measured slot 1 had two in 200000 shots, flipping 74 logical qubits.
# Refuted: indexed Z36 binomial gap spectra selected chiefly by
# component cycles consistently give zero Z failures: the current
# five measured slots had respectively 0, 1, 4, 3 and 1 Z failures
# in 200000 shots apiece, despite zero X failures in all five.
# Refuted: adding parity tiers and prioritizing verified HX traps on
# the parent's shapes consistently improves its five-code score:
# fitness was 7.2168 versus 7.2252 for its direct parent.
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
