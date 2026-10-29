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
CONCEPT = ("C6 x C2 x C2, order 24, A=(1,3), B=(2,7): common-annihilator A trinomials and screened degree-two/three B columns should produce five distance-six readings and fitness above 15.8 in cell 0.450.")
GROUP = ("Keep the C_m x C2 x C2 construction rule. The parent reached fitness 48.354 at order 44, but the new size factor 23 requires q<=26. Order 24 is this family's largest fitting order, giving n=552. Its C3 x C2^3 decomposition permits deliberate common-kernel control; transfer of the parent's distance performance is not assumed.")
SHAPES = ("A=(1,3), B=(2,7) give component lengths/check counts 72/24 and 168/48, n=552, kb=240 and guaranteed rate 10/23. Verified ranks retain k=250..262, entirely inside cell 0.450. Parent minima occupied sector 1; slot 34's weight-22 X family used Acol0/Acol1 at Bcol1. Replace both component structures rather than relabel those blocks.")
FILLING = ("Twenty recipes cross five B support architectures with four binomial-relative-order rules. A trinomials use all C3 residues and three distinct C2^2 cosets, targeting a common kernel of dimension four. Both component row totals are nine; B column totals are two or three, so both CSS weights are <=12. Verified lifts screen rates, component dependencies through weight five, sampled kernel witnesses and full logicals through weight four. Permutation-invariant fingerprints separate emitted codes; readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r045_C6xC2xC2_13x27_socle_triples_kernel5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C6 x C2 x C2 requires q=24")
    # Same direct-product construction rule; only m and the order change.
    m = 6
    els = [(a, v & 1, v >> 1) for v in range(4) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % 2, g[2] % 2)

    def mul(g, h):
        return ((g[0] + h[0]) % m,
                (g[1] + h[1]) % 2,
                (g[2] + h[2]) % 2)

    def inv(g):
        return ((-g[0]) % m, g[1] % 2, g[2] % 2)

    def idx(g):
        a, b, c = red(g)
        return a + m * (b + 2 * c)

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Column pairs (top,bottom). Every plan has two row totals of nine,
# three degree-two columns and four degree-three columns.
# Zero cells are deliberate, but no row or column is dead.
_B_COLUMN_PLANS = (
    ((2, 1), (2, 1), (1, 2), (1, 2), (1, 1), (1, 1), (1, 1)),
    ((2, 1), (1, 2), (1, 2), (1, 2), (2, 0), (1, 1), (1, 1)),
    ((2, 1), (2, 1), (1, 2), (1, 2), (2, 0), (0, 2), (1, 1)),
    ((3, 0), (2, 1), (1, 2), (1, 2), (0, 2), (1, 1), (1, 1)),
    ((3, 0), (0, 3), (2, 1), (1, 2), (1, 1), (1, 1), (1, 1)),
)

from functools import lru_cache


def _column_syndromes(rows, n):
    """Columns of the verified matrix, represented as syndrome integers."""
    cols = [0] * n
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= marker
            row ^= bit
    return cols


def _four_cycle_count(cols):
    """Exact Tanner four-cycle count from pairs of incident checks."""
    overlaps = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for j, b in enumerate(checks):
            for a in checks[:j]:
                key = (a, b)
                old = overlaps.get(key, 0)
                cycles += old
                overlaps[key] = old + 1
    return cycles


def _has_light_logical(cols, stabilizer_rows):
    """Exhaustively detect nontrivial kernel words of weights 1..4.

    Collisions among supports of sizes 0, 1 and 2 cover every such
    kernel word. Reduction modulo the stabilizer span distinguishes
    logical operators from stabilizers.
    """
    basis = {}
    for row in stabilizer_rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    n = len(cols)
    bits = [1 << j for j in range(n)]
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is not None:
            if nontrivial(bits[j] ^ old):
                return True
        else:
            seen[syndrome] = bits[j]

    for j in range(n):
        sj, bj = cols[j], bits[j]
        for i in range(j):
            syndrome = sj ^ cols[i]
            word = bj ^ bits[i]
            old = seen.get(syndrome)
            if old is not None:
                if nontrivial(word ^ old):
                    return True
            else:
                seen[syndrome] = word
    return False


def _component_columns(hx, hz, q):
    """Extract both component actions only from the VERIFIED LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            packed = 0
            for j in range(ca):
                packed |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(packed)
    bm = (1 << (cb * q)) - 1
    br = [source & bm for source in hz[:rb * q]]
    return (_column_syndromes(ar, ca * q),
            _column_syndromes(br, cb * q))


def _component_floor(columns, through_five=False):
    """Exact classical-kernel minimum through four or five.

    The return values five/six mean that all smaller weights were
    excluded, not that that returned weight has a witness. These
    component tests do not certify the full quantum distance.
    """
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2

    pairs = set()
    found_four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found_four = True
            pairs.add(syndrome)
    if found_four:
        return 4
    if not through_five:
        return 5

    # With no dependencies through four, a triple/pair collision
    # necessarily has disjoint supports and detects weight five.
    for i, x in enumerate(columns):
        for j in range(i):
            partial = x ^ columns[j]
            for h in range(j):
                if partial ^ columns[h] in pairs:
                    return 5
    return 6


def _kernel_reading(rng, columns, trials):
    """Sample actual component-kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        # Elementary reductions retain actual kernel vectors.
        words.sort(key=int.bit_count)
        short = words[:36]
        for i in range(len(short)):
            word = short[i]
            weight = word.bit_count()
            for other in short[:i]:
                z = word ^ other
                zw = z.bit_count()
                if 0 < zw < weight:
                    word, weight = z, zw
            short[i] = word
            best = min(best, weight)
        for i, word in enumerate(short):
            for other in short[:i]:
                z = word ^ other
                if z:
                    best = min(best, z.bit_count())
        if best <= 5:
            break
    return best


def _incidence_fingerprint(hx, hz, cx, cz):
    """Invariant under qubit/check permutations and a global X/Z swap.

    Unequal fingerprints prove nonisomorphism. Equal fingerprints
    are conservatively rejected even when the codes might differ.
    """
    def participation(columns):
        incident = []
        counts = {}
        for syndrome in columns:
            active = []
            while syndrome:
                lo = syndrome & -syndrome
                active.append(lo.bit_length() - 1)
                syndrome ^= lo
            pairs = [(a, b) for j, b in enumerate(active)
                     for a in active[:j]]
            incident.append(pairs)
            for pair in pairs:
                counts[pair] = counts.get(pair, 0) + 1
        return [sum(counts[pair] - 1 for pair in pairs)
                for pairs in incident]

    px, pz = participation(cx), participation(cz)
    normal = tuple(sorted(
        (x.bit_count(), z.bit_count(), a, b)
        for x, z, a, b in zip(cx, cz, px, pz)))
    swapped = tuple(sorted(
        (z.bit_count(), x.bit_count(), b, a)
        for x, z, a, b in zip(cx, cz, px, pz)))
    checks = tuple(sorted((
        tuple(sorted(row.bit_count() for row in hx)),
        tuple(sorted(row.bit_count() for row in hz)))))
    return checks, min(normal, swapped)


def _draw_architecture(rng, G, plan, mode):
    """Internal relative-label rules, not global orbit transformations."""
    A = [[]]
    for j in range(3):
        # The even C6 residues form C3. Distinct V4 projections make
        # each triple a radical element at the nontrivial C3 character.
        # Independent triples target their shared socle, rather than
        # three gauge copies of one polynomial.
        residues = rng.sample(range(3), 3)
        cosets = rng.sample(range(4), 3)
        gauge = G.pick(rng)
        A[0].append([
            G.mul(gauge, (2 * a, v & 1, v >> 1))
            for a, v in zip(residues, cosets)
        ])

    cyclic_six = [(a, 0, 0) for a in (1, 5)]
    mixed_six = [
        (a, v & 1, v >> 1)
        for a in (1, 2, 4, 5) for v in range(1, 4)
    ]
    cyclic_three = [(2, 0, 0), (4, 0, 0)]
    B = [[], []]
    for j, column in enumerate(plan):
        for i, count in enumerate(column):
            if count == 0:
                cell = []
            elif count == 1:
                cell = [G.pick(rng)]
            elif count == 2:
                # A binomial in a one-row-only column must not have
                # order-three or involution separation.
                pure = column[1 - i] == 0
                if pure:
                    pool = mixed_six
                elif mode == 0:
                    pool = cyclic_six
                elif mode == 1:
                    pool = mixed_six
                elif mode == 2:
                    pool = cyclic_three
                else:
                    pool = cyclic_three if (i + j) & 1 else mixed_six
                first = G.pick(rng)
                cell = [first, G.mul(first, rng.choice(pool))]
            else:
                # Repeated C3 residue plus one different residue has
                # nonzero evaluation at both C3 character classes.
                # Thus these pure-row triples avoid the deliberately
                # singular architecture used on A.
                phase = rng.randrange(3)
                residues = (phase, phase, (phase + 1) % 3)
                cosets = rng.sample(range(4), 3)
                cell = [
                    (a + 3 * rng.randrange(2), v & 1, v >> 1)
                    for a, v in zip(residues, cosets)
                ]
            assert len(cell) == len(set(cell))
            B[i].append(cell)
    return A, B


@lru_cache(maxsize=1)
def _design_bank(G):
    """Deterministic, jointly deduplicated bank; caching changes no outputs."""
    bank = []
    fingerprints = set()

    for slot in range(N_SLOTS):
        recipe = slot % 20
        plan = _B_COLUMN_PLANS[recipe // 4]
        mode = recipe % 4
        sa = [[3, 3, 3]]
        sb = [[column[i] for column in plan] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(45024061 + 100003 * slot)

        def merit(item):
            d = min(6, item["da"], item["db"])
            return (min(item["fa"], item["fb"]), d,
                    item["k"] * d * d, -item["cycles"])

        pool = []
        for attempt in range(48):
            A, B = _draw_architecture(rng, G, plan, mode)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # 250/552=.453 and 262/552=.475 minus a small margin.
            # Exact ranks, not the common-kernel hypothesis, decide admission.
            if not 250 <= k <= 262:
                continue
            ac, bc = _component_columns(hx, hz, G.q)
            fa, fb = _component_floor(ac), _component_floor(bc)
            da = fa if fa < 5 else min(6, _kernel_reading(rng, ac, 4))
            db = fb if fb < 5 else min(6, _kernel_reading(rng, bc, 4))
            item = {
                "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
                "ac": ac, "bc": bc, "fa": fa, "fb": fb,
                "da": da, "db": db,
                "cycles": _four_cycle_count(ac) + _four_cycle_count(bc),
            }
            pool.append(item)
            pool.sort(key=merit, reverse=True)
            del pool[8:]

        if not pool:
            bank.append(None)
            continue

        for item in pool:
            if item["fa"] == 5:
                item["da"] = min(
                    item["da"], _kernel_reading(rng, item["ac"], 16))
            if item["fb"] == 5:
                item["db"] = min(
                    item["db"], _kernel_reading(rng, item["bc"], 16))
        pool.sort(key=merit, reverse=True)

        winner = None
        for item in pool:
            # Deepen the best component proposals independently.
            da, db = item["da"], item["db"]
            if da >= 6:
                da = min(da, _component_floor(item["ac"], True))
            if db >= 6:
                db = min(db, _component_floor(item["bc"], True))

            hx, hz, n = item["hx"], item["hz"], item["n"]
            cx = _column_syndromes(hx, n)
            cz = _column_syndromes(hz, n)
            fingerprint = _incidence_fingerprint(hx, hz, cx, cz)
            if fingerprint in fingerprints:
                continue
            clean = (not _has_light_logical(cx, hz)
                     and not _has_light_logical(cz, hx))
            d = min(6, da, db)
            objective = (
                clean, d, item["k"] * d * d, -item["cycles"])
            if winner is None or objective > winner[0]:
                winner = (objective, item, fingerprint)
            # All admissible codes are in the target cell. A clean
            # component-six finalist is sufficient; do not optimize
            # lucky larger classical readings.
            if clean and d == 6:
                break

        if winner is None:
            bank.append(None)
            continue
        _, item, fingerprint = winner
        assert css_ok(item["hx"], item["hz"])
        assert 425 * item["n"] <= 1000 * item["k"] < 475 * item["n"]
        fingerprints.add(fingerprint)
        bank.append((item["A"], item["B"]))

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a prescribed recipe's independently optimized fresh filling."""
    bank = _design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("recipe found no invariant-distinct target-rate filling")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the weight cap with rolled uniform fillings alone does not eliminate weight-4 logicals: all five measured parent codes had d30s=4, with minimum operators in sector 2.
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
