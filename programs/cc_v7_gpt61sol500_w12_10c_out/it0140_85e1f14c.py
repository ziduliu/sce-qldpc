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
CONCEPT = ("For cyclic Z23, order 23, encoded as Z1:Z23, with A=B=(3,4), planting a common degree-eleven Golay-factor rank deficiency while retaining rank-one augmentation should yield k=79 at n=575 and five measured distance readings >=16.")
GROUP = ("Keep the cyclic metacyclic rule and order 23. RUN STATE records fitness 29.5 on the parent's asymmetric bed and 27.3 on Z23 (3,4)x(3,4). The factorization of x^23+1 into the augmentation factor and two degree-eleven Golay factors supplies a new rank mechanism; it does not establish a distance advantage.")
SHAPES = ("Target cell 0.150 with equal (3,4) sides: n=575, kb=23, guaranteed rate 1/25, and two 92-variable/69-check components. Rank-one augmentation contributes 12 surplus logical dimensions; a common rank-two component specialization at one degree-eleven factor contributes another 44, predicting k=79 and rate 79/575. Verified component ranks require 56 and full CSS ranks require k=79. Parent minima broadly occupied sector-1 Acol0-2/Bcol0-4, predominantly at Z weight 15, so replace both component geometries.")
FILLING = ("Twenty cards cross five A support profiles with four B profiles. Sparse cells satisfy a planted polynomial row relation modulo the binary Golay factor; field arithmetic chooses labels only, while every screened matrix comes from lifted_matrices. Each augmentation column is (1,1,0), removing single-block constant-vector kernels. Profiles have maximum CSS weight twelve and distinct cell terms. Screen both component kernels through six, seed valid constant-block pairs, independently rescreen full-code witnesses, and conservatively reject equal strengthened Tanner invariants. Fresh deterministic seeds avoid replaying archived fillings. Witness readings remain upper bounds.")
FAMILY_NAME = "r015_m1r23s1_34x34_golayfactor_aug1_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 4)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


from functools import lru_cache


def _golay_mul(a, b):
    """Field arithmetic for choosing labels, NOT a lifted matrix."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 2048:
            # x^11+x^9+x^7+x^6+x^5+x+1.
            a ^= 0xAE3
    return out


@lru_cache(maxsize=1)
def _golay_label_data():
    """Perfect Golay syndrome catalogue, checked before using its labels."""
    from itertools import combinations

    powers = [1]
    for _ in range(22):
        powers.append(_golay_mul(powers[-1], 2))
    assert len(set(powers)) == 23
    assert _golay_mul(powers[-1], 2) == 1

    small = {0: 0}
    for weight in (1, 2, 3):
        for support in combinations(range(23), weight):
            syndrome, word = 0, 0
            for b in support:
                syndrome ^= powers[b]
                word |= 1 << b
            # This verifies uniqueness through weight three rather than
            # relying on an untested generator-polynomial convention.
            assert syndrome not in small
            small[syndrome] = word
    assert len(small) == 2048

    quads = {}
    for support in combinations(range(23), 4):
        syndrome, word = 0, 0
        for b in support:
            syndrome ^= powers[b]
            word |= 1 << b
        quads.setdefault(syndrome, []).append(word)

    def power(a, exponent):
        out = 1
        while exponent:
            if exponent & 1:
                out = _golay_mul(out, a)
            a = _golay_mul(a, a)
            exponent >>= 1
        return out

    inverses = [0] + [power(a, 2046) for a in range(1, 2048)]
    assert all(_golay_mul(a, inverses[a]) == 1
               for a in range(1, 2048))
    pairs = tuple((s, word) for s, word in small.items()
                  if word.bit_count() == 2)
    return tuple(powers), small, quads, tuple(inverses), pairs


# Column count profiles. Every augmentation column is (1,1,0).
# Their row totals are respectively (4,4,8), (6,4,6), (6,6,4),
# (6,6,6), and (6,6,6); column maxima are four or six.
_SPECTRAL_PROFILES = (
    ((1,1,2), (1,1,2), (1,1,2), (1,1,2)),
    ((3,1,0), (1,1,2), (1,1,2), (1,1,2)),
    ((3,1,0), (1,3,0), (1,1,2), (1,1,2)),
    ((3,1,2), (1,3,0), (1,1,2), (1,1,2)),
    ((1,1,4), (3,1,0), (1,3,0), (1,1,2)),
)


def _spectral_grid(rng, G, profile):
    """Plant p0 + U*p1 + V*p2 = 0 at one Golay factor.

    Common column translates are gauge freedom, not new designs, and
    are deliberately omitted. Distinct normalized columns change the
    actual polynomial matrix.
    """
    assert G.q == 23
    powers, small, quads, inverses, pairs = _golay_label_data()
    normal_count = profile.count((1,1,2))

    for _ in range(512):
        U = rng.randrange(1, 2048)
        V = rng.randrange(1, 2048)
        ui, vi = inverses[U], inverses[V]

        if (3,1,0) in profile and small[U].bit_count() != 3:
            continue
        if (1,3,0) in profile and small[ui].bit_count() != 3:
            continue

        normal = []
        for b, value in enumerate(powers):
            syndrome = _golay_mul(1 ^ _golay_mul(U, value), vi)
            word = small[syndrome]
            if word.bit_count() == 2:
                normal.append((1, 1 << b, word))
        if len(normal) < normal_count:
            continue

        special = {}
        if (3,1,2) in profile:
            choices = []
            for syndrome, word in pairs:
                first = small[U ^ _golay_mul(V, syndrome)]
                if first.bit_count() == 3:
                    choices.append((first, 1, word))
            if not choices:
                continue
            special[(3,1,2)] = rng.choice(choices)

        if (1,1,4) in profile:
            choices = []
            for b, value in enumerate(powers):
                syndrome = _golay_mul(1 ^ _golay_mul(U, value), vi)
                for word in quads.get(syndrome, ()):
                    choices.append((1, 1 << b, word))
            if not choices:
                continue
            special[(1,1,4)] = rng.choice(choices)

        selected = iter(rng.sample(normal, normal_count))
        columns = []
        for counts in profile:
            if counts == (1,1,2):
                column = next(selected)
            elif counts == (3,1,0):
                column = (small[U], 1, 0)
            elif counts == (1,3,0):
                column = (1, small[ui], 0)
            else:
                column = special[counts]
            assert tuple(word.bit_count() for word in column) == counts
            columns.append(column)

        if len(set(columns)) != 4:
            continue
        grid = [
            [[(0, b) for b in range(23) if columns[j][i] >> b & 1]
             for j in range(4)]
            for i in range(3)
        ]
        return grid
    raise ValueError("no sparse Golay-factor row relation for this profile")


def _EVEN_SLOT(rng, G, idx):
    """Legacy helper name; now optimize a planted spectral-rank card.
    Every matrix used for screening comes from lifted_matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1

    card = idx % 20
    ap = _SPECTRAL_PROFILES[card // 4]
    bp = _SPECTRAL_PROFILES[card % 4]
    sa = [[column[i] for column in ap] for i in range(ra)]
    sb = [[column[i] for column in bp] for i in range(rb)]
    assert q == 23 and A_SHAPE == B_SHAPE == (3, 4)
    assert gf2_rank([
        sum((count & 1) << j for j, count in enumerate(row))
        for row in sa]) == 1
    assert gf2_rank([
        sum((count & 1) << j for j, count in enumerate(row))
        for row in sb]) == 1
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def columns(rows, length):
        out = [0] * length
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def components(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        # Extract beta=0 from HX and j=0 from HZ, preserving the
        # verified lift's actions. These are the two component matrices.
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
        return hx, hz, n, columns(ar, ca * q), columns(br, cb * q)

    def four_cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                t = (x & cols[j]).bit_count()
                total += t * (t - 1) // 2
        return total

    def short_kernel(cols):
        """Return a kernel witness of weight <=6, or None.
        Collisions among all subsets of size <=3 cover every such word."""
        seen = {0: 0}

        def insert(s, v):
            old = seen.get(s)
            if old is not None:
                return old ^ v
            seen[s] = v
            return None

        length = len(cols)
        for i in range(length):
            hit = insert(cols[i], 1 << i)
            if hit is not None:
                return hit
        for i in range(length):
            for j in range(i):
                hit = insert(cols[i] ^ cols[j], (1 << i) | (1 << j))
                if hit is not None:
                    return hit
        for i in range(length):
            for j in range(i):
                s = cols[i] ^ cols[j]
                v = (1 << i) | (1 << j)
                for h in range(j):
                    hit = insert(s ^ cols[h], v | (1 << h))
                    if hit is not None:
                        return hit
        return None

    def row_basis(rows):
        basis = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = row
                    break
                row ^= old
        return basis

    def outside_span(v, basis):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    def logical_estimate(cols, opposite_basis, side):
        """Sample component kernel words, rejecting LP stabilizers.
        The returned positive weight is an upper bound from actual witnesses."""
        best = [ca * cb * q + ra * rb * q + 1]

        def consider(v):
            w = v.bit_count()
            if not v or w >= best[0]:
                return
            if side == "A":
                for beta in range(cb):
                    full = 0
                    for j in range(ca):
                        full |= ((v >> (j * q)) & mask) << (
                            (j * cb + beta) * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return
            else:
                for j in range(ca):
                    full = v << (j * cb * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return

        # Augmentation columns are identical and nonzero: constant
        # single blocks are NOT kernels, but pairs of constant blocks are.
        # Check their actual syndromes before counting an embedding.
        width = ca if side == "A" else cb
        for block in range(width):
            for other in range(block):
                word = ((mask << (block * q)) |
                        (mask << (other * q)))
                syndrome = 0
                support = word
                while support:
                    low = support & -support
                    syndrome ^= cols[low.bit_length() - 1]
                    support ^= low
                assert syndrome == 0
                consider(word)

        order = list(range(len(cols)))
        rounds = 28
        for _ in range(rounds):
            rng.shuffle(order)
            pivots = {}
            words = []
            for i in order:
                s, v = cols[i], 1 << i
                while s:
                    p = s.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (s, v)
                        break
                    s ^= old[0]
                    v ^= old[1]
                if not s:
                    words.append(v)
                    consider(v)
            # Shortening preserves the exact component kernel. These
            # vectors need not remain an independent basis for probing.
            for sweep in range(2):
                rng.shuffle(words)
                for i, v in enumerate(words):
                    weight = v.bit_count()
                    for j, other in enumerate(words):
                        if i == j:
                            continue
                        trial = v ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            v, weight = trial, tw
                    words[i] = v
                    consider(v)
            words.sort(key=int.bit_count)
            for i, x in enumerate(words):
                for j in range(i):
                    consider(x ^ words[j])
            small = words[:12]
            for i, x in enumerate(small):
                for j in range(i):
                    xy = x ^ small[j]
                    for h in range(j):
                        consider(xy ^ small[h])
            if best[0] <= 8:
                break
        return best[0] if best[0] <= ca * cb * q + ra * rb * q else 0

    def mixed_reading(hx, hz, n, ceiling):
        """Search both full CSS kernels, including mixed-sector words.
        The ceiling already comes from an actual component logical."""
        best = ceiling
        for checks, stabilizers in ((hx, hz), (hz, hx)):
            cols = columns(checks, n)
            span = row_basis(stabilizers)
            order = list(range(n))

            def consider(v):
                nonlocal best
                if v and v.bit_count() < best and outside_span(v, span):
                    best = v.bit_count()

            for _ in range(12):
                rng.shuffle(order)
                pivots, words = {}, []
                for i in order:
                    s, v = cols[i], 1 << i
                    while s:
                        p = s.bit_length() - 1
                        old = pivots.get(p)
                        if old is None:
                            pivots[p] = (s, v)
                            break
                        s ^= old[0]
                        v ^= old[1]
                    if not s:
                        words.append(v)
                        consider(v)
                words.sort(key=int.bit_count)
                for i, v in enumerate(words):
                    for other in words[:i]:
                        consider(v ^ other)
                for start in words[:12]:
                    v = start
                    for sweep in range(3):
                        old = v
                        weight = v.bit_count()
                        for other in words:
                            trial = v ^ other
                            tw = trial.bit_count()
                            if 0 < tw < weight:
                                v, weight = trial, tw
                        consider(v)
                        if v == old:
                            break
                if len(words) >= 3:
                    for trial in range(192):
                        a, b, c = rng.sample(words, 3)
                        consider(a ^ b ^ c)
                if best <= 8:
                    break
        return best

    winner = None
    reserve = None
    finalists = []
    accepted = 0

    for attempt in range(64):
        # Independent sparse row relations, not independent mutations:
        # a free cell mutation would destroy the planted spectral rank.
        A = _spectral_grid(rng, G, ap)
        B = _spectral_grid(rng, G, bp)
        hx, hz, n, ac, bc = components(A, B)
        cycle = [four_cycles(ac), four_cycles(bc)]
        k = n - gf2_rank(hx) - gf2_rank(hz)

        # Augmentation rank one, Golay-factor rank two, and the other
        # degree-eleven rank three give component rank 1+22+33=56.
        # Full verified ranks also check that the two defects align.
        if (n != 575 or k != 79
                or gf2_rank(ac) != 56 or gf2_rank(bc) != 56):
            continue

        short_a, short_b = short_kernel(ac), short_kernel(bc)
        wa = short_a.bit_count() if short_a is not None else 7
        wb = short_b.bit_count() if short_b is not None else 7
        reserve_key = (min(wa, wb), wa + wb, -sum(cycle), k)
        if reserve is None or reserve_key > reserve[0]:
            reserve = (reserve_key, A, B)
        if short_a is not None or short_b is not None:
            continue

        dz = logical_estimate(ac, row_basis(hz), "A")
        dx = logical_estimate(bc, row_basis(hx), "B")
        if not dz or not dx:
            continue
        d = min(dx, dz)
        key = (k * d * d / n, d, dx + dz, -sum(cycle))
        finalists.append((key, A, B, d, k))
        if winner is None or key > winner[0]:
            winner = (key, A, B)
        accepted += 1
        if accepted >= 8:
            break

    # Fresh component searches and full-kernel searches can only lower
    # each finalist's retained upper reading.
    refined = []
    for key, A, B, d, k in sorted(
            finalists, key=lambda item: item[0], reverse=True)[:3]:
        hx, hz, n, ac, bc = components(A, B)
        dz = logical_estimate(ac, row_basis(hz), "A")
        dx = logical_estimate(bc, row_basis(hx), "B")
        if not dz or not dx:
            continue
        d = mixed_reading(hx, hz, n, min(d, dz, dx))
        refined.append(((k * d * d / n, d, dx + dz, key[-1]), A, B))

    chosen = (max(refined, key=lambda item: item[0]) if refined
              else winner if winner is not None else reserve)
    if chosen is None:
        raise ValueError("no positive-k candidate in the requested rate band")
    A, B = chosen[1], chosen[2]
    hx, hz, n = lifted_matrices(G, A, B)
    assert n == 575
    assert n - gf2_rank(hx) - gf2_rank(hz) == 79
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B}}


from functools import lru_cache
from collections import Counter


def _even_graph_fingerprint(hx, hz, n):
    """Invariant of the combined, untyped Tanner graph.
    Unequal fingerprints prove nonisomorphism. Equality is discarded
    conservatively; it is not claimed to prove isomorphism."""
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    overlap = Counter()
    for i, row in enumerate(rows):
        wi = weights[i]
        for j in range(i):
            wj = weights[j]
            overlap[(min(wi, wj), max(wi, wj),
                     (row & rows[j]).bit_count())] += 1

    neighborhoods = [[] for _ in range(n)]
    incidence = [0] * n
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            neighborhoods[c].append(weights[i])
            incidence[c] |= 1 << i
            row ^= bit
    local = Counter(tuple(sorted(v)) for v in neighborhoods)

    # Triangle profiles of the untyped check-intersection graph add a
    # longer-walk invariant beyond degrees and pairwise overlaps.
    adjacency = []
    for i, row in enumerate(rows):
        neighbors = 0
        while row:
            low = row & -row
            neighbors |= incidence[low.bit_length() - 1]
            row ^= low
        adjacency.append(neighbors & ~(1 << i))

    walks = Counter()
    for i, neighbors in enumerate(adjacency):
        triangles = 0
        degree_profile = Counter()
        remaining = neighbors
        while remaining:
            low = remaining & -remaining
            j = low.bit_length() - 1
            triangles += (neighbors & adjacency[j]).bit_count()
            degree_profile[(weights[j], adjacency[j].bit_count())] += 1
            remaining ^= low
        walks[(weights[i], neighbors.bit_count(), triangles,
               tuple(sorted(degree_profile.items())))] += 1

    return (tuple(sorted(weights)),
            tuple(sorted(overlap.items())),
            tuple(sorted(local.items())),
            tuple(sorted(walks.items())))


@lru_cache(maxsize=1)
def _even_design_bank(G):
    """Pure deterministic bank; caching only avoids repeating its search."""
    bank, used = [], set()
    for slot in range(N_SLOTS):
        chosen = None
        for retry in range(4):
            local_rng = random.Random(
                BASE_SEED * 10007 + 57515079 + slot * 100003 + retry * 7919)
            try:
                ctx = _EVEN_SLOT(local_rng, G, slot)
            except ValueError:
                continue
            A, B = ctx["cells"]["A"], ctx["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            fingerprint = _even_graph_fingerprint(hx, hz, n)
            if fingerprint in used:
                continue
            used.add(fingerprint)
            chosen = (A, B)
            break
        # A failed recipe does not invalidate successful bank entries.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign one freshly optimized, invariant-distinct filling."""
    chosen = _even_design_bank(G)[idx]
    if chosen is None:
        raise ValueError("no invariant-distinct target-rate filling")
    A, B = chosen
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return every new-shape cell without subsequent randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled, predominantly singleton A fillings reliably suppress weight-six Z witnesses; rejected by the parent: four of five measured codes have Z weight 6, while X witnesses weigh 16-20.
# REFUTED: Degree-four A columns, four-cycle optimization, and exclusion through component weight six suffice for five measured distances >=14 on Z23 (2,4)x(3,5); the measured parent instead returned four distances of 12 and one of 13.
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
