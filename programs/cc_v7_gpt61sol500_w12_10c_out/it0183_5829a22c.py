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
CONCEPT = ("Cyclic C16, order 16, shapes (2,5)x(2,6): an even-augmentation A row with verified component ranks 31/32 should retain five distance-eight readings with k=196, exceeding the 0.350 frontier fitness of 23.0310.")
GROUP = ("Jump from S3 x C4 to cyclic C16, a genuinely different group and an unlisted bed. The related C17 cubic-component bed measured five [578,208,8] codes. C16's repeated-root group algebra may change both ranks and short words, so neither rank surplus nor distance transfers without screening.")
SHAPES = ("Target cell 0.350 rather than the parent's distance-four 0.550 plateau. Shapes (2,5)x(2,6) give n=544, kb=192 and guaranteed rate 12/34. Components have 80/96 variables and 32 checks each. Require verified component ranks 31/32 and assembled k=196, rate 0.360294. Parent minima broadly occupied sector-1 Acol/Bcol blocks; replace its degree-two components with cubic columns.")
FILLING = ("Twenty cards cross three B incidence profiles with seven canonical A gap geometries. A column counts are four (2,1) and one (0,3); B columns are cubic with row totals nine. Thus HX<=11 and HZ<=12. Optimize actual labels using verified component lifts, exact kernel screening through six, independent kernel-witness rescreening, and four-cycle tie-breaks. Require assembled rank and CSS checks; uncoloured Tanner invariants separate emitted codes. Component floors are not quantum-distance certificates.")
FAMILY_NAME = "r035_C16_25x26_evenrow_rank31_cubic6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C16 requires order 16")
    els = [(a,) for a in range(16)]

    def red(g):
        return (g[0] % 16,)

    def mul(g, h):
        return ((g[0] + h[0]) % 16,)

    def inv(g):
        return ((-g[0]) % 16,)

    def idx(g):
        return g[0] % 16

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


from collections import Counter
from functools import lru_cache
from itertools import combinations

_C16_ACOUNTS = ((2, 1),) * 4 + ((0, 3),)
_C16_BCOUNTS = (
    ((2, 1),) * 3 + ((1, 2),) * 3,
    ((3, 0), (2, 1)) + ((1, 2),) * 4,
    ((3, 0), (0, 3)) + ((2, 1),) * 2 + ((1, 2),) * 2,
)


@lru_cache(maxsize=1)
def _c16_geometries():
    """Unsigned gap sets modulo all C16 automorphisms and column order."""
    geometries = set()
    for ds in combinations(range(1, 8), 4):
        canonical = min(
            tuple(sorted(min((u*d) % 16, (-u*d) % 16)
                         for d in ds))
            for u in range(1, 16, 2))
        geometries.add(canonical)

    def triangles(ds):
        return sum(any((a + sign*b) % 16 in (c, (-c) % 16)
                       for sign in (-1, 1))
                   for a, b, c in combinations(ds, 3))

    out = sorted(geometries, key=lambda ds: (triangles(ds), ds))
    if len(out) < 7:
        raise ValueError("fewer than seven distinct gap geometries")
    return tuple(out)


def _c16_component(G, side, cells):
    """Extract components exclusively from the scaffold's verified lift."""
    q = G.q
    mask = (1 << q) - 1
    zero_a = [[[] for _ in range(5)] for _ in range(2)]
    zero_b = [[[] for _ in range(6)] for _ in range(2)]
    if side == "A":
        hx, _, _ = lifted_matrices(G, cells, zero_b)
        rows = [
            sum(((hx[(i*6)*q+t] >> (j*6*q)) & mask) << (j*q)
                for j in range(5))
            for i in range(2) for t in range(q)]
        width = 5*q
    else:
        _, hz, _ = lifted_matrices(G, zero_a, cells)
        width = 6*q
        rows = [row & ((1 << width)-1) for row in hz[:2*q]]

    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    assert all(c.bit_count() == 3 for c in cols)
    return rows, cols


def _c16_floor(cols, through_six=False):
    """Exact component-kernel floor through four or six.

    Cubic columns exclude odd-weight kernel words. Equal syndromes
    among supports of size <=3 cover every even kernel weight <=6.
    Sentinels six/eight mean only absence through the screened range.
    """
    if 0 in cols:
        return 1
    if len(set(cols)) != len(cols):
        return 2
    pairs = set()
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            syndrome = ci ^ cj
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)
    if not through_six:
        return 6

    singles = set(cols)
    triples = set()
    for i, ci in enumerate(cols):
        for j in range(i):
            pair = ci ^ cols[j]
            for k in range(j):
                syndrome = pair ^ cols[k]
                if syndrome in singles or syndrome in triples:
                    return 6
                triples.add(syndrome)
    return 8


def _c16_cycles(rows):
    return sum(
        ((a & b).bit_count() * ((a & b).bit_count()-1)) // 2
        for i, a in enumerate(rows) for b in rows[:i])


def _c16_probe(cols, rng, trials):
    """Search true classical kernel witnesses; readings are upper bounds."""
    order = list(range(len(cols)))
    memory = []
    best = len(cols)+1
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome and word:
                words.append(word)

        words = sorted(set(words+memory), key=lambda v: (v.bit_count(), v))
        for sweep in range(2):
            changed = False
            for i in range(len(words)):
                word, weight = words[i], words[i].bit_count()
                for other in words[:i]:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        word, weight = trial, tw
                        changed = True
                words[i] = word
            words.sort(key=lambda v: (v.bit_count(), v))
            if not changed:
                break

        short = sorted(set(words+memory),
                       key=lambda v: (v.bit_count(), v))[:32]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    discoveries.append(trial)
        memory = sorted(set(discoveries),
                        key=lambda v: (v.bit_count(), v))[:24]
        best = min(best, min((v.bit_count() for v in memory),
                             default=len(cols)+1))
        if best <= 4:
            break
    return best, tuple(memory)


def _c16_draw(rng, counts, gaps=None):
    """Column gauges remove translations, not distinct relative designs."""
    columns = []
    for j, (top, bottom) in enumerate(counts):
        x = rng.sample(range(16), top)
        y = rng.sample(range(16), bottom)
        if gaps is not None and j < 4:
            x = [0, gaps[j]]
        anchor = x[0] if x else y[0]
        columns.append((
            sorted((v-anchor) % 16 for v in x),
            sorted((v-anchor) % 16 for v in y)))
    return [[[(v,) for v in columns[j][i]]
             for j in range(len(counts))] for i in range(2)]


def _c16_replace(rng, grid, counts, gaps, positions):
    proposal = _c16_draw(rng, counts, gaps)
    out = [[list(cell) for cell in row] for row in grid]
    for j in positions:
        for i in range(2):
            out[i][j] = proposal[i][j]
    return out


def _c16_optimize(G, side, counts, gaps, rng):
    """Optimize one prescribed component architecture, then rescreen."""
    required_rank = 31 if side == "A" else 32
    pool, labels = [], set()

    def admit(grid):
        signature = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if signature in labels:
            return
        labels.add(signature)
        rows, cols = _c16_component(G, side, grid)
        if gf2_rank(rows) != required_rank:
            return
        if _c16_floor(cols) < 6:
            return
        reading, words = _c16_probe(cols, rng, 3)
        cycles = _c16_cycles(rows)
        load = sum(word.bit_count() <= 8 for word in words)
        key = (min(reading, 8), -load, -cycles)
        pool.append(dict(grid=grid, rows=rows, cols=cols, words=words,
                         reading=reading, cycles=cycles, key=key))
        pool.sort(key=lambda item: item["key"], reverse=True)
        del pool[8:]

    for attempt in range(112):
        admit(_c16_draw(rng, counts, gaps))

    for step in range(56):
        if not pool:
            admit(_c16_draw(rng, counts, gaps))
            continue
        parent = pool[rng.randrange(min(4, len(pool)))]
        short = parent["words"][:6]
        pressure = [
            1 + sum(((word >> (16*j)) & 65535).bit_count()
                    for word in short)
            for j in range(len(counts))]
        j = rng.choices(range(len(counts)), weights=pressure, k=1)[0]
        positions = {j}
        if step % 5 == 0:
            positions.add(rng.randrange(len(counts)))
        admit(_c16_replace(rng, parent["grid"], counts, gaps, positions))

    if not pool:
        return None

    for item in pool:
        reading, words = _c16_probe(item["cols"], rng, 12)
        item["reading"] = min(item["reading"], reading)
        item["words"] = tuple(sorted(set(item["words"]+words),
                                    key=lambda v: (v.bit_count(), v))[:24])
        item["key"] = (
            min(item["reading"], 8),
            -sum(w.bit_count() <= 8 for w in item["words"]),
            -item["cycles"])
    pool.sort(key=lambda item: item["key"], reverse=True)

    finalists = []
    for item in pool[:5]:
        floor = _c16_floor(item["cols"], through_six=True)
        reading, words = _c16_probe(item["cols"], rng, 32)
        item["reading"] = min(item["reading"], reading)
        item["floor"] = floor
        item["key"] = (
            min(floor, item["reading"], 8),
            -sum(w.bit_count() <= 8 for w in words),
            -item["cycles"])
        finalists.append(item)
    return max(finalists, key=lambda item: item["key"])


def _c16_small_logical(checks, stabilizers, n):
    """Exact stabilizer-aware rejection of logical weights one through four."""
    basis = {}
    for source in stabilizers:
        while source:
            p = source.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = source
                break
            source ^= old

    def outside(word):
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    cols = [0]*n
    for r, source in enumerate(checks):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit

    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif outside(word ^ old):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif outside(word ^ old):
                return True
    return False


def _c16_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariants; different values certify nonisomorphism."""
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length()-1].append(r)
            source ^= bit
    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = Counter()
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] += multiplicity*other
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            sum(neighbors.get(t, 0)*value for t, value in square.items()),
            tuple(sorted(Counter(square.values()).items()))))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _c16_bank(G):
    """Each slot owns a support/gap card; never use an orbit-copy fallback."""
    assert G.q == 16
    geometries = _c16_geometries()
    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        card = slot % 20
        geometry = (card//3 + 7*(slot//20)) % len(geometries)
        gaps = geometries[geometry]
        bc = _C16_BCOUNTS[card % 3]
        sa = [[c[i] for c in _C16_ACOUNTS] for i in range(2)]
        sb = [[c[i] for c in bc] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        selected = None

        for batch in range(3):
            local = random.Random(350160196 + 100019*slot + 7919*batch)
            a = _c16_optimize(G, "A", _C16_ACOUNTS, gaps, local)
            b = _c16_optimize(G, "B", bc, None, local)
            if a is None or b is None:
                continue
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 544 or k != 196:
                continue
            if _c16_small_logical(hx, hz, n):
                continue
            if _c16_small_logical(hz, hx, n):
                continue
            fingerprint = _c16_fingerprint(hx, hz, n)
            if fingerprint in fingerprints:
                continue
            assert css_ok(hx, hz)
            assert 325*n <= 1000*k < 375*n
            assert max(row.bit_count() for row in hx+hz) <= W_MAX
            assert all(len(cell) == len(set(cell)) <= 5
                       for grid in (A, B) for row in grid for cell in row)
            A = tuple(tuple(tuple(cell) for cell in row) for row in A)
            B = tuple(tuple(tuple(cell) for cell in row) for row in B)
            selected = (A, B, a["floor"], b["floor"])
            fingerprints.add(fingerprint)
            break
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the card's rank-verified, incidence-distinct filling."""
    selected = _c16_bank(G)[idx]
    if selected is None:
        raise ValueError("no new rank-196 filling passed this card's screening")
    A, B, af, bf = selected
    return {"cells": {"A": A, "B": B}, "k": 196,
            "component_kernel_floors": (af, bf)}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without altering its screened labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen odd-central deletion slots supply fifteen inequivalent S3 x C4 codes: ten principal slots collapsed to isomorphic duplicates, and the five measured survivors all retained distance-four readings.
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
