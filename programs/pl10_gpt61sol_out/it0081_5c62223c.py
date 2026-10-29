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
CONCEPT = ("On C37, order 37, shapes (1,3)x(1,5), spectrally distinct triangle-free binomial component pairs should improve worse-basis circuit reliability over the parent's triangle-prone direction cards at rate approximately 0.514.")
GROUP = ("C37 is genuinely different from S3 x C5 and is an untried bed in the supplied BEDS. Order 37 is the largest lift allowed by size factor 16; every nonidentity direction has order 37. The recorded C35 high-rate bed scored about 3.5, supporting a cyclic experiment without proving a group advantage.")
SHAPES = ("Keep (1,3)x(1,5): n=592, guaranteed kb=296 and rate 0.500; components have 111 and 185 variables and 37 checks each. Connected binomial fillings are verified for k=304, rate 0.51351. Parent failures affected essentially all 248 logicals across many sector-one Acol/Bcol blocks with approximately 1.0-1.1x enrichment, not a localized Brow defect.")
FILLING = ("Enumerate triangle-free inverse-normalized generator sets, quotient multiplier images, and retain different exact component graph spectra. Different spectral pairs distinguish the main slots even under arbitrary full Tanner relabelings: check degrees 8/12 and data degrees 4/16 identify the component subgraphs. Optimize relative embeddings using verified full lifts and stabilizer-aware weight-three/four logical and possible-hook counts. Main cards have row weights 8/12 and 16872 six-round CNOTs. Larger harnesses may extend to explicit trinomial/unit support architectures. These costs test suppression of the parent's 135-204 Z failures per 250 shots; they are not pL measurements or ASC hook certificates.")
FAMILY_NAME = "r050_C37_13x15_spectral_trianglefree_hookcards"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This prime cyclic bed requires q=37")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


def _columns(rows, width):
    """Extract column syndromes from the scaffold's verified matrices."""
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            if c < width:
                out[c] |= tag
            row ^= bit
    return out


def _spectral_book(G, width, side):
    """Different exact graph spectra, not independently drawn orbit copies."""
    from itertools import combinations

    q = G.q
    half = (q - 1) // 2
    representatives = set()

    for directions in combinations(range(1, half + 1), width):
        signed = set(directions) | {q - d for d in directions}
        # Necessary algebraic generation filter only. The actual graph
        # and its triangle count are subsequently obtained from the lift.
        if any((a + b) % q in signed for a in signed for b in signed):
            continue
        canonical = min(
            tuple(sorted(min((u * d) % q, (-u * d) % q)
                         for d in directions))
            for u in range(1, half + 1)
        )
        representatives.add(canonical)

    spectra = {}
    for directions in sorted(representatives):
        cells = [[[G.e, (d,)] for d in directions]]
        if side == "A":
            rows = lifted_matrices(G, cells, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], cells)[1]

        edges = _columns(rows, width * q)
        if (gf2_rank(rows) != q - 1 or
                len(set(edges)) != len(edges) or
                any(edge.bit_count() != 2 for edge in edges)):
            continue

        adjacency = [0] * q
        for edge in edges:
            bit = edge & -edge
            a = bit.bit_length() - 1
            b = (edge ^ bit).bit_length() - 1
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

        triangles = squares = 0
        for a in range(q):
            for b in range(a + 1, q):
                common = (adjacency[a] & adjacency[b]).bit_count()
                squares += common * (common - 1) // 2
                if (adjacency[a] >> b) & 1:
                    triangles += common
        if triangles:
            continue
        squares //= 2

        neighbors = []
        for mask in adjacency:
            ids = []
            while mask:
                bit = mask & -mask
                ids.append(bit.bit_length() - 1)
                mask ^= bit
            neighbors.append(ids)

        # These verified component graphs are vertex-transitive.
        # q times each return count is an adjacency spectral moment.
        # Moments through q determine the full characteristic polynomial.
        walks = [1] + [0] * (q - 1)
        moments = []
        for length in range(1, q + 1):
            walks = [sum(walks[v] for v in ids) for ids in neighbors]
            moments.append(walks[0])
        signature = tuple(moments)
        entry = (squares, signature, directions)
        old = spectra.get(signature)
        if old is None or entry < old:
            spectra[signature] = entry

    book = sorted(spectra.values())
    if not book:
        raise ValueError("triangle-free spectral catalogue is empty")
    return book


def _short_hook_profile(checks, stabilizers, n):
    """Exact logical counts through weight four; hooks are only a proxy."""
    syndromes = _columns(checks, n)
    spread = _columns(stabilizers, n)
    if 0 in syndromes or len(set(syndromes)) != n:
        return None

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Linear normal forms modulo the stabilizer row space.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        row = pivots[p] ^ (1 << p)
        residue = 0
        while row:
            bit = row & -row
            residue ^= residues[bit.bit_length() - 1]
            row ^= bit
        residues[p] = residue

    singles = dict(zip(syndromes, residues))
    pairs = {}
    triples = quartets = hook_one = hook_two = 0

    for a in range(n):
        sa, ra, ha = syndromes[a], residues[a], spread[a]
        for b in range(a):
            syndrome = sa ^ syndromes[b]
            residue = ra ^ residues[b]
            third = singles.get(syndrome)
            if third is not None and residue != third:
                triples += 1

            hook = (ha & spread[b]).bit_count()
            old = pairs.get(syndrome)
            if old is None:
                pairs[syndrome] = (residue, hook)
                continue

            if isinstance(old, tuple):
                previous, previous_hook = old
                different = int(residue != previous)
                quartets += different
                hook_one += different * (hook + previous_hook)
                hook_two += different * hook * previous_hook
                if different:
                    classes = {
                        previous: [1, previous_hook],
                        residue: [1, hook],
                    }
                else:
                    classes = {residue: [2, hook + previous_hook]}
                pairs[syndrome] = [2, hook + previous_hook, classes]
            else:
                same = old[2].get(residue)
                same_count, same_hooks = same if same else (0, 0)
                different = old[0] - same_count
                different_hooks = old[1] - same_hooks
                quartets += different
                hook_one += hook * different + different_hooks
                hook_two += hook * different_hooks
                old[0] += 1
                old[1] += hook
                if same is None:
                    old[2][residue] = [1, hook]
                else:
                    same[0] += 1
                    same[1] += hook

    # Distinct nonzero syndromes make colliding different pairs disjoint.
    # Every triple/quartet has three pair partitions. Hook sums retain
    # those partitions to count possible spread opportunities.
    return triples // 3, quartets // 3, hook_one, hook_two


def _same_basis_four(rows):
    return sum(
        overlap * (overlap - 1) // 2
        for i, row in enumerate(rows)
        for other in rows[:i]
        for overlap in [(row & other).bit_count()]
    )


def NEW_CTX(rng, G, idx):
    """Choose a distinct spectral pair, then optimize its relative embedding."""
    abook = _spectral_book(G, A_SHAPE[1], "A")
    bbook = _spectral_book(G, B_SHAPE[1], "B")
    pairs = []
    for a, aa in enumerate(abook):
        for b, bb in enumerate(bbook):
            # Component cycles can appear across five A copies or three
            # B copies. This minimax cost is not a circuit prediction.
            az, bx = 5 * aa[0], 3 * bb[0]
            pairs.append((max(az, bx), az + bx, abs(az - bx), a, b))
    pairs.sort()

    extension = idx >= len(pairs)
    if not extension:
        a, b = pairs[idx][-2:]
        architecture = ("spectral_pair", a, b)
    else:
        # Only needed if a larger harness exhausts the binary catalogue.
        # These are actual support changes, not extra automorphism slots.
        extra = idx - len(pairs)
        pair_number, profile = divmod(extra, 3)
        if pair_number >= len(pairs):
            raise ValueError("deliberate design catalogue exhausted")
        a, b = pairs[pair_number][-2:]
        architecture = ("trinomial_unit_extension", a, b, profile)

    ad, bd = abook[a][2], bbook[b][2]
    units = list(range(1, (G.q + 1) // 2))
    rng.shuffle(units)
    units = [1] + [u for u in units if u != 1]
    trials = 24 if extension else min(6, len(units))
    seen = set()
    best = None

    for trial in range(trials):
        u = units[trial % len(units)]
        A = [[[G.e, (d,)] for d in ad]]
        transformed = sorted(min((u * d) % G.q, (-u * d) % G.q)
                             for d in bd)
        B = [[[G.e, (d,)] for d in transformed]]

        if extension:
            B[0][0] = [G.e]
            for j in range(2 - profile, 3):
                alternatives = [g for g in G.elements if g not in A[0][j]]
                A[0][j].append(rng.choice(alternatives))

        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row
        )
        if signature in seen:
            continue
        seen.add(signature)

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if _grid_weights(sa, sb) > min(12, W_MAX):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 19 * n <= 40 * k < 21 * n):
            continue
        if not extension and k != 8 * G.q + 8:
            continue

        px = _short_hook_profile(hz, hx, n)
        pz = _short_hook_profile(hx, hz, n)
        if px is None or pz is None:
            continue
        if not extension and (px[0] or pz[0]):
            continue

        score = tuple(
            value for t in (0, 2, 3, 1)
            for value in (max(px[t], pz[t]), px[t] + pz[t])
        )
        fx, fz = _same_basis_four(hx), _same_basis_four(hz)
        score += (max(fx, fz), fx + fz)
        if best is None or score < best[0]:
            best = (score, A, B, hx, hz, k)

    if best is None:
        raise ValueError("spectral card failed verified full-CSS screening")
    score, A, B, hx, hz, k = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert _no_dead_lines(
        [[len(cell) for cell in row] for row in A],
        [[len(cell) for cell in row] for row in B])
    return {"idx": idx, "architecture": architecture,
            "A": A, "B": B, "k": k, "risk": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without subsequent randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal d2s and row-weight cap do not imply comparable circuit scores: parent slots 4 and 52 both had d2s=10,w=12, but s=6.000 versus 5.019 with strong basis imbalance.
# REFUTED: Long-order binomial directions and triangle/square priorities suffice for reliable S3 x C5 memory: fitness was 3.1183, measured Z failures were 135-204/250, and weight-three/four sector-one logicals remained abundant.
# REFUTED: Twenty direction-count cards reliably produce twenty different Tanner designs: the parent produced only 11 distinct codes across 60 slots, with extensive isomorphic deduplication.
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
