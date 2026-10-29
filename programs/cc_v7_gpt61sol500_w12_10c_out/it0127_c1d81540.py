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
CONCEPT = ("Dic13, order 52, shapes (1,3)x(1,3): replacing involutory reflections with order-four dicyclic cosets and screening order-controlled trinomials should produce five measured distance readings at least ten at exact rate 0.400.")
GROUP = ("Jump from dihedral D29 to Dic13. Related dicyclic beds lead cell 0.100 and have reached distance twelve near rate 0.300, motivating but not proving transfer. Dic13 has a unique involution, a normal prime-order C13 subgroup and a C4 quotient; its prescribed order 52 gives n=520.")
SHAPES = ("Retain equal (1,3) sides for cell 0.400: n=520, guaranteed k>=208 and equal 156-variable/52-check components. Require full component row ranks and verify exact k=208. The parent's thousands of weight-eight minima occupied sector-1 Acol/Bcol blocks, prominently Acol1 and Bcol1.")
FILLING = ("Twenty architectural cards cross order-13/order-26 rotation-difference populations with all-mixed versus one rotation-only cell. All cells have three distinct terms and both maximum check weights are twelve. Optimize verified component-kernel witness readings; deepen finalists independently, screen complete kernels through weight four and probe mixed three-plus-three supports. Joint selection rejects repeated uncolored Tanner fingerprints. Component and bounded mixed-sector probes are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r040_dic13_13x13_ordercards_distinct_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 52

def custom_ops(q):
    # Dic13: x^26=1, y^2=x^13, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic13 requires order 52")
    els = [(a, b) for b in range(2) for a in range(26)]

    def red(g):
        a, b = g
        return ((a + 13 * (b // 2)) % 26, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 13*b*d) % 26,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 26, 0) if b == 0 else ((a + 13) % 26, 1)

    def idx(g):
        a, b = red(g)
        return 26*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


from collections import Counter
from functools import lru_cache


_D13_POPULATIONS = tuple((a, b) for a in range(4) for b in range(a, 4))


def _d13_columns(rows, width):
    """Transpose rows obtained exclusively from the verified constructor."""
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= marker
            row ^= low
    return cols


def _d13_components(hx, hz, q):
    mask = (1 << q) - 1
    ar = [
        sum(((row >> (3*j*q)) & mask) << (j*q)
            for j in range(3))
        for row in hx[:q]
    ]
    br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
    return ar, br


def _d13_free4(cols):
    """Exact exclusion through four for these odd-weight columns."""
    if any(c.bit_count() % 2 != 1 for c in cols):
        return False
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    pairs = set()
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            if s in pairs:
                return False
            pairs.add(s)
    return True


def _d13_has6(cols):
    """Exact six-support existence after successful free4 screening.

    Distinct colliding triples must then be disjoint. A zero result
    exhausts the triple test; a hit supplies an actual component word.
    """
    seen = set()
    for i in range(2, len(cols)):
        for j in range(1, i):
            xy = cols[i] ^ cols[j]
            for k in range(j):
                s = xy ^ cols[k]
                if s in seen:
                    return True
                seen.add(s)
    return False


def _d13_cycles(cols):
    pairs = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            low = col & -col
            neighbors.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                old = pairs.get(key, 0)
                total += old
                pairs[key] = old + 1
    return total


def _d13_probe(cols, rng, trials):
    """Actual nonzero classical kernel witnesses, not distance proofs."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
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
                words.append(word)
                best = min(best, word.bit_count())

        # Basis replacement preserves nonzero kernel vectors and span.
        for sweep in range(2):
            rng.shuffle(words)
            changed = False
            for i, word in enumerate(words):
                weight = word.bit_count()
                for j, other in enumerate(words):
                    if i == j:
                        continue
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        word, weight = trial, tw
                        changed = True
                words[i] = word
                best = min(best, weight)
            if not changed:
                break

        short = sorted(set(words), key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    best = min(best, trial.bit_count())
        if best <= 4:
            break
    return best


def _d13_mixed6(cols, q, x_checks):
    """Search exactly the 3-tail + 3-sector-1 support class.

    This does NOT exhaust all mixed six-supports. Rejecting a found
    stabilizer as well as a logical is deliberately conservative.
    """
    n1 = 9*q
    mask = (1 << q) - 1
    singles = [set() for _ in range(3)]
    for c in range(n1):
        block = (c // q) % 3 if x_checks else (c // q) // 3
        singles[block].add((cols[c] >> (block*q)) & mask)
    tail = [
        tuple((col >> (b*q)) & mask for b in range(3))
        for col in cols[n1:]
    ]
    for i in range(q):
        for j in range(i):
            xy = tuple(tail[i][b] ^ tail[j][b] for b in range(3))
            for k in range(j):
                if all((xy[b] ^ tail[k][b]) in singles[b]
                       for b in range(3)):
                    return True
    return False


def _d13_draw(rng, G, slot):
    """Order populations and pure-cell presence are architectural cards."""
    card = slot % 20
    pa, pb = _D13_POPULATIONS[card % 10]
    pure = card // 10
    lane = slot // 20
    grids = []
    for side, population in enumerate((pa, pb)):
        row = []
        for j in range(3):
            repeated = j < population
            # Even nonzero deltas have order 13. Odd deltas here have
            # order 26; x^13, the unique involution, is never selected.
            position = (card + 2*j + 3*side + lane*(j+1)) % 6
            delta = 2*(position+1) if repeated else 2*position+1
            delta = (delta * rng.choice((-1, 1))) % 26
            if pure and side == 0 and j == 0:
                choices = [
                    a for a in range(1, 26)
                    if a not in (delta, 13)
                    and (a-delta) % 26 != 13
                ]
                third = (rng.choice(choices), 0)
            else:
                third = (rng.randrange(26), 1)
            cell = [G.e, (delta, 0), third]
            assert len(set(cell)) == 3
            assert all(G.mul(G.inv(g), h) != (13, 0)
                       for i, g in enumerate(cell) for h in cell[:i])
            row.append(cell)
        grids.append([row])
    return grids


def _d13_fingerprint(hx, hz, n):
    """Permutation invariant of the uncolored combined Tanner graph.

    The weighted check-intersection graph is induced by that graph.
    Its local two-step walk profiles and qubit degrees are invariant
    under arbitrary qubit/check permutations and global X/Z exchange.
    Different fingerprints therefore certify nonisomorphic fillings.
    Equal fingerprints are conservatively rejected, not declared iso.
    """
    rows = hx + hz
    adjacency = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            shared = (row & rows[j]).bit_count()
            if shared:
                adjacency[i][j] = shared
                adjacency[j][i] = shared
    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = {}
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] = square.get(t, 0) + multiplicity*other
        triangles = sum(value*square.get(s, 0)
                        for s, value in neighbors.items())
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0), triangles,
            tuple(sorted(Counter(square.values()).items())),
        ))
    degrees = Counter(c.bit_count() for c in _d13_columns(rows, n))
    return (tuple(sorted(degrees.items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _d13_bank(G):
    """Joint deterministic selection makes every emitted slot distinct."""
    assert G.q == 52 and A_SHAPE == B_SHAPE == (1, 3)
    fingerprints, bank = set(), []

    def merit(item):
        d = min(item["da"], item["db"], 16)
        return (item["clean"], d*d, d,
                min(item["da"], 20) + min(item["db"], 20),
                -item["cycles"])

    for slot in range(N_SLOTS):
        rng = random.Random(BASE_SEED*10007 + 130052 + 100003*slot)
        chosen = None
        seen_cells = set()

        # Fresh batches search the card's labels; they do not transform
        # previously selected codes by automorphisms or permutations.
        for batch in range(3):
            pool = []
            for attempt in range(48):
                A, B = _d13_draw(rng, G, slot)
                label_key = tuple(tuple(sorted(cell))
                                  for grid in (A, B)
                                  for row in grid for cell in row)
                if label_key in seen_cells:
                    continue
                seen_cells.add(label_key)
                hx, hz, n = lifted_matrices(G, A, B)
                ar, br = _d13_components(hx, hz, G.q)
                if gf2_rank(ar) != G.q or gf2_rank(br) != G.q:
                    continue
                ac = _d13_columns(ar, 3*G.q)
                bc = _d13_columns(br, 3*G.q)
                ca, cb = _d13_free4(ac), _d13_free4(bc)
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
                    "ac": ac, "bc": bc, "ca": ca, "cb": cb,
                    "clean": int(ca and cb),
                    "da": _d13_probe(ac, rng, 3),
                    "db": _d13_probe(bc, rng, 3),
                    "cycles": _d13_cycles(ac) + _d13_cycles(bc),
                }
                pool.append(item)
                pool.sort(key=merit, reverse=True)
                del pool[6:]

            for item in pool:
                for side, clean in (("a", item["ca"]), ("b", item["cb"])):
                    cols = item[side+"c"]
                    if clean and _d13_has6(cols):
                        item["d"+side] = min(item["d"+side], 6)
                    else:
                        item["d"+side] = min(
                            item["d"+side], _d13_probe(cols, rng, 16))
                xc = _d13_columns(item["hx"], item["n"])
                zc = _d13_columns(item["hz"], item["n"])
                item["clean"] = int(
                    _d13_free4(xc) and _d13_free4(zc))
                if item["clean"]:
                    if (_d13_mixed6(xc, G.q, True)
                            or _d13_mixed6(zc, G.q, False)):
                        item["da"] = min(item["da"], 6)
                        item["db"] = min(item["db"], 6)

            # Independent rescreening can only lower retained readings.
            pool.sort(key=merit, reverse=True)
            for item in pool[:3]:
                item["da"] = min(
                    item["da"], _d13_probe(item["ac"], rng, 48))
                item["db"] = min(
                    item["db"], _d13_probe(item["bc"], rng, 48))

            for item in sorted(pool, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert n == 520 and k == 208
                fingerprint = _d13_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                chosen = (item, fingerprint)
                break
            if chosen is not None:
                break

        if chosen is None:
            # Never substitute an orbit copy to populate a failed slot.
            bank.append(None)
            continue
        item, fingerprint = chosen
        assert css_ok(item["hx"], item["hz"])
        assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
        assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
        fingerprints.add(fingerprint)
        bank.append({"A": item["A"], "B": item["B"]})

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return one jointly optimized, invariant-distinct architectural card."""
    selected = _d13_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh exact-rate filling for this architectural card")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled counts with uniform labels suffice to escape short sector-1 logicals at rate 0.400; many seed slots have d2s=2, and all five measured seed codes have d30s=4 with sector-1 minima.
# REFUTED: Canonical reflection-gap schedules alone prevent Tanner-isomorphic duplicates; the current parent emitted only 35 distinct codes from 60 slots despite canonical gap profiles.
# REFUTED: The shown dihedral mixed-trinomial short-kernel screens reliably exceed measured distance eight; their five measured leaders all retained d30s=8 with thousands of sector-1 weight-eight witnesses.
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
