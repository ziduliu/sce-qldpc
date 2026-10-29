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
CONCEPT = ("Nonabelian C8 semidirect C4 with action 5, order 32, shapes (1,4)x(1,4): reducing repeated-cell weight-four vulnerabilities at check weight 10 should improve worse-basis circuit reliability over C35 without requiring a larger distance.")
GROUP = ("The action satisfies 5^4=1 modulo 8. This group has centre order eight and is genuinely different from C35 and the recorded order-32 bed with centre two. Noncommuting generators can remove abelian commutator squares; no circuit advantage is asserted before measurement.")
SHAPES = ("Keep equal 1x4 components, each with 32 checks and 128 variables. n=544 and kb=288 guarantee rate 9/17 in target cell 0.550; connected all-binomial fillings have measured-rank target k=296. Parent failures and weight-four witnesses spread over sector-one Acol/Bcol blocks with mostly 1.0-1.1x enrichment, so reinforce both components rather than an unproven hotspot.")
FILLING = ("Enumerate inverse-normalized non-involutory generator sets modulo actual group automorphisms, retaining different colored short-word and graph-walk signatures. Slots select distinct unordered component-brief pairs, balancing weight-four multiplicity with repeated-cell pairs as a possible hook-risk proxy. All-binomial fillings use check weight 10 and 15360 six-round CNOTs versus the measured parent's 16800-18900. Catalogue extensions use explicit independently searched degree profiles without fourth-order heavy cells. Final verified lifts screen ranks, short logicals, CSS and both weight sums. The target is the parent's 154/250 all-binomial X failures and 180-203/250 heavy-B X failures, not distance as fitness.")
FAMILY_NAME = "r055_c8sd5c4_14x14_coloredword_catalogue_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C8 semidirect C4 with action 5 requires q=32")
    # x^8=y^4=1, y*x*y^-1=x^5; elements are x^a*y^b.
    m, r, s = 8, 4, 5
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _short_logical_counts(checks, stabilizers, n):
    """Exact counts of nontrivial logical supports of weights 1..4.
    Matrices come exclusively from the verified scaffold lift."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(v):
        while v:
            p = v.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            v ^= old
        return False

    columns = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= syndrome_bit
            row ^= bit

    singles = {}
    counts = [0, 0, 0, 0]
    bits = [1 << a for a in range(n)]
    for a, syndrome in enumerate(columns):
        singles.setdefault(syndrome, []).append(a)
        if syndrome == 0 and nontrivial(bits[a]):
            counts[0] += 1

    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = columns[a] ^ columns[b]
            support = bits[a] | bits[b]
            if syndrome == 0 and nontrivial(support):
                counts[1] += 1

            # a < b < c counts each weight-three support once.
            for c in singles.get(syndrome, ()):
                if c > b and nontrivial(support | bits[c]):
                    counts[2] += 1

            bucket = pairs.get(syndrome)
            if bucket is None:
                pairs[syndrome] = [(a, b, support)]
            else:
                # c < d < a < b is the unique ordered partition
                # used to count each weight-four support once.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        counts[3] += 1
                bucket.append((a, b, support))
    return tuple(counts)


def _single_row_components(hx, hz, q):
    """Project components only from the scaffold's verified lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((hx[t] >> (j * 4 * q)) & mask) << (j * q)
            for j in range(4))
        for t in range(q)
    ]
    br = [hz[t] & ((1 << (4 * q)) - 1) for t in range(q)]
    return ar, br


def _binary_brief(rows, q):
    """Colored weight-four words and uncolored graph invariants.

    A binary component is a graph incidence matrix. Everything here
    is extracted from verified rows, not from a separate group lift.
    Repeated-cell pairs are only a possible hook-risk surrogate.
    """
    columns = [0] * (4 * q)
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    if (len(set(columns)) != len(columns) or
            any(s.bit_count() != 2 for s in columns)):
        return None
    if gf2_rank(rows) != q - 1:
        return None

    adjacency = [0] * q
    for syndrome in columns:
        low = syndrome & -syndrome
        a = low.bit_length() - 1
        b = (syndrome ^ low).bit_length() - 1
        adjacency[a] |= 1 << b
        adjacency[b] |= 1 << a

    # Each triangle is counted on its three edges.
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        later = neighbors >> (a + 1)
        while later:
            bit = later & -later
            b = a + 1 + bit.bit_length() - 1
            triangles += (neighbors & adjacency[b]).bit_count()
            later ^= bit
    if triangles:
        return None

    patterns = ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
    words = {p: 0 for p in patterns}
    pairs = {}
    for a in range(len(columns)):
        for b in range(a + 1, len(columns)):
            syndrome = columns[a] ^ columns[b]
            bucket = pairs.setdefault(syndrome, [])
            for c, d in bucket:
                if d >= a:
                    continue
                colors = {}
                for t in (c, d, a, b):
                    color = t // q
                    colors[color] = colors.get(color, 0) + 1
                partition = tuple(sorted(colors.values(), reverse=True))
                words[partition] += 1
            bucket.append((a, b))

    counts = tuple(words[p] for p in patterns)
    total = sum(counts)
    repeated_pairs = sum(
        number * sum(t * (t - 1) // 2 for t in partition)
        for partition, number in zip(patterns, counts)
    )

    seen = frontier = 1
    shells = []
    while frontier:
        shells.append(frontier.bit_count())
        next_frontier = 0
        bits = frontier
        while bits:
            bit = bits & -bits
            next_frontier |= adjacency[bit.bit_length() - 1]
            bits ^= bit
        next_frontier &= ~seen
        seen |= next_frontier
        frontier = next_frontier
    assert seen == (1 << q) - 1

    neighbor_lists = []
    for neighbors in adjacency:
        ids = []
        while neighbors:
            bit = neighbors & -neighbors
            ids.append(bit.bit_length() - 1)
            neighbors ^= bit
        neighbor_lists.append(ids)
    walks = [1] + [0] * (q - 1)
    returns = []
    for length in range(1, 9):
        walks = [sum(walks[t] for t in ids) for ids in neighbor_lists]
        if length >= 4:
            returns.append(walks[0])

    signature = (counts, tuple(shells), tuple(returns))
    # Heuristic weighting, not a calibrated circuit failure probability.
    risk = 2 * total + 3 * repeated_pairs
    return risk, total, repeated_pairs, signature


def _generator_book(G):
    """Enumerate component briefs, quotienting actual automorphisms.

    No slot is made by applying an automorphism to a saved filling.
    Different signatures are conservative design distinctions; the
    evaluator still performs its stronger full Tanner-isomorphism test.
    """
    from itertools import combinations

    q, els, e = G.q, G.elements, G.e

    def powers(g, count):
        out = [e]
        for _ in range(count):
            out.append(G.mul(out[-1], g))
        return out

    inverse_class = [
        min(t, G.idx(G.inv(g))) for t, g in enumerate(els)
    ]
    generators = [
        t for t, g in enumerate(els)
        if inverse_class[t] == t and G.mul(g, g) != e
    ]

    # Every automorphism is determined by images of x and y. Enumerate
    # homomorphisms satisfying the presentation, then require bijectivity.
    automorphisms = set()
    for x in els:
        xp = powers(x, 8)
        if xp[8] != e or xp[4] == e:
            continue
        for y in els:
            yp = powers(y, 4)
            if yp[4] != e:
                continue
            if G.mul(G.mul(y, x), G.inv(y)) != xp[5]:
                continue
            image = tuple(
                G.idx(G.mul(xp[a], yp[b]))
                for b in range(4) for a in range(8)
            )
            if len(set(image)) == q:
                automorphisms.add(image)
    assert automorphisms
    automorphisms = sorted(automorphisms)

    seen_orbits = set()
    briefs = {}
    for quad in combinations(generators, 4):
        if quad in seen_orbits:
            continue
        orbit = {
            tuple(sorted(inverse_class[permutation[t]] for t in quad))
            for permutation in automorphisms
        }
        seen_orbits.update(orbit)
        canonical = min(orbit)
        A = [[[e, els[t]] for t in canonical]]
        # The zero opposite component is a probe, never a submitted code.
        rows = lifted_matrices(G, A, [[[]]])[0]
        brief = _binary_brief(rows, q)
        if brief is None:
            continue
        entry = brief + (canonical,)
        signature = brief[3]
        old = briefs.get(signature)
        if old is None or entry < old:
            briefs[signature] = entry
    book = sorted(briefs.values())
    if not book:
        raise ValueError("no connected triangle-free binary brief")
    return book


def _same_basis_four(rows):
    return sum(
        overlap * (overlap - 1) // 2
        for i, row in enumerate(rows)
        for other in rows[:i]
        for overlap in [(row & other).bit_count()]
    )


def NEW_CTX(rng, G, idx):
    """Assign a classified component pair, not another optimized orbit."""
    from itertools import combinations_with_replacement

    book = _generator_book(G)
    pairs = []
    for a, b in combinations_with_replacement(range(len(book)), 2):
        aa, bb = book[a], book[b]
        key = (max(aa[0], bb[0]), aa[0] + bb[0],
               max(aa[1], bb[1]), aa[1] + bb[1],
               max(aa[2], bb[2]), aa[2] + bb[2], a, b)
        pairs.append(key)
    pairs.sort()

    def binary(number):
        return [[[G.e, G.elements[t]] for t in book[number][4]]]

    pool = []
    if idx < len(pairs):
        pair = pairs[idx]
        a, b = pair[-2:]
        A, B = binary(a), binary(b)
        hx, hz, n = lifted_matrices(G, A, B)
        pool.append((pair[:-2], 0, A, B, hx, hz, n))
        design = ("binary_briefs", a, b)
    else:
        # Continue to actual degree-profile changes if the classified
        # binary catalogue cannot supply the harness's requested slots.
        extension = idx - len(pairs)
        profiles = (
            ((2, 2, 2, 2), (2, 2, 2, 3)),
            ((2, 2, 2, 3), (2, 2, 2, 3)),
            ((2, 2, 2, 2), (2, 2, 3, 3)),
            ((2, 2, 2, 2), (1, 3, 3, 3)),
        )
        profile = extension % len(profiles)
        embedding = extension // len(profiles)
        if embedding >= len(pairs):
            raise ValueError("deliberate component design catalogue exhausted")
        a, b = pairs[embedding][-2:]
        ac, bc = profiles[profile]
        assert _grid_weights([list(ac)], [list(bc)]) <= W_MAX
        nonidentity = [g for g in G.elements if g != G.e]

        def extend(number, counts):
            grid = binary(number)
            for j, weight in enumerate(counts):
                if weight == 1:
                    grid[0][j] = [G.e]
                elif weight == 3:
                    for _ in range(128):
                        u, v = rng.sample(nonidentity, 2)
                        if G.mul(u, v) != G.mul(v, u):
                            grid[0][j] = [G.e, u, v]
                            break
                    else:
                        raise ValueError("noncommuting trinomial unavailable")
            return grid

        seen = set()
        for attempt in range(32):
            A, B = extend(a, ac), extend(b, bc)
            identity = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row
            )
            if identity in seen:
                continue
            seen.add(identity)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _single_row_components(hx, hz, G.q)
            aa = _short_logical_counts(ar, [], 4 * G.q)
            bb = _short_logical_counts(br, [], 4 * G.q)
            key = (aa[0] + bb[0], aa[1] + bb[1],
                   max(aa[2], bb[2]), aa[2] + bb[2],
                   max(aa[3], bb[3]), aa[3] + bb[3])
            pool.append((key, attempt, A, B, hx, hz, n))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[3:]
        design = ("degree_extension", profile, a, b)

    best = None
    for component_key, attempt, A, B, hx, hz, n in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0 or 40 * k <= 21 * n:
            continue
        if design[0] == "binary_briefs":
            assert k == 9 * G.q + 8
        x = _short_logical_counts(hz, hx, n)
        z = _short_logical_counts(hx, hz, n)
        cx, cz = _same_basis_four(hx), _same_basis_four(hz)
        edges = sum(row.bit_count() for row in hx + hz)
        key = (x[0] + z[0], x[1] + z[1],
               max(x[2], z[2]), x[2] + z[2],
               max(x[3], z[3]), x[3] + z[3],
               max(cx, cz), cx + cz, edges, component_key, attempt)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k)

    if best is None:
        raise ValueError("no positive-k target-cell filling")
    key, A, B, hx, hz, k = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert css_ok(hx, hz)
    return {"idx": idx, "case": design, "k": k,
            "screen": key, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
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
