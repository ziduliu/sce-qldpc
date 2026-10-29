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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dicyclic Dic2=Q8, order 8, A=B=(2,7): triangle-free regular and singly grounded incidence constituents, separated only after coupling, should supply at least five novel target-cell codes with certified distance four.")
GROUP = ("Keep the dicyclic construction rule x^(2t)=1, y^2=x^t, y*x*y^-1=x^-1. Factor 53 excludes the parent's order 28; t=2 gives order eight and n=424. Q8 has three non-involutory inverse classes. The parent's thirteen distinct codes all measured distance three; changing order alone is not assumed to improve distance.")
SHAPES = ("Equal constituents have 56 variables and 16 checks. n=424, kb=200 and guaranteed rate 25/53=0.471698 lie just below cell 0.500; verified ranks must give 0.475<=k/n<0.525. Regular pairs are prioritized, targeting k=212 and score eight; singly grounded pairs may provide additional target-rate classes. Parent weight-three minima occupied sector-1 Acol/Bcol blocks. Two-row incidence geometry replaces the parent's monomial rank pivots.")
FILLING = ("Enumerate balanced internal-edge/matching supports and six-matchings plus one grounding singleton. Constituent rows total at most seven and columns at most two, so BOTH CSS maxima are at most nine, also below twelve. Keep deck-labelled descriptions until full uncoloured Tanner intersection and intrinsic high-degree projection walk screening. Verified lifts exclude constituent and complete commuting supports through three, enforce actual target-cell ranks and CSS, and verify a non-stabilizer weight-four witness. Unequal invariants certify within-program separation; archive novelty and five-code yield remain hypotheses.")
FAMILY_NAME = "r050_dicyclic_q8_27x27_incidence_full3_certified4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2=Q8 requires order eight")
    # Same dicyclic rule, now t=2:
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1.
    t = 2
    m = 2*t
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1)*c + t*b*d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + t) % m, b)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


_DIC_PROFILES = (
    (4, 4, 4), (4, 4, 7), (4, 4, 14), (4, 7, 7),
    (4, 7, 14), (4, 14, 14), (7, 7, 7), (7, 7, 14),
    (7, 14, 14), (14, 14, 14),
)
_DIC_PRIMARY = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 4), (0, 9), (1, 5), (1, 8), (2, 3),
    (2, 7), (3, 5), (4, 6), (5, 8), (6, 9),
)
_DIC_CARDS = _DIC_PRIMARY + tuple(
    (a, b) for a in range(10) for b in range(a, 10)
    if (a, b) not in _DIC_PRIMARY
)


def _dic_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid finite-group order")


def _dic_component(rng, G, orders, pools):
    """A fixed monomial pivot plus three prescribed-order binomials."""
    row = [[G.e]]
    used = set()
    for order in orders:
        available = [h for h in pools[order]
                     if h not in used and G.inv(h) not in used]
        if not available:
            raise ValueError("relative-order card exhausted")
        h = rng.choice(available)
        used.add(h)
        g = G.pick(rng)
        row.append([g, G.mul(g, h)])
    return [row]


def _dic_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_excludes_two(rows, n):
    """Complete exclusion of commuting supports of weights one and two."""
    columns = _dic_columns(rows, n)
    return 0 not in columns and len(set(columns)) == n


def _dic_three_witness(G, B):
    """X witness in Acol1: one B-binomial bit and two B-pivot bits.

    B[0][0]=identity. For binomial labels u,v, selecting its column
    at identity has check syndromes at u^-1 and v^-1, canceled by
    those two columns of the monomial pivot.
    """
    q = G.q
    u, v = B[0][1]
    base = 4*q                  # Acol1/Bcol0
    return ((1 << (base + q + G.idx(G.e)))
            | (1 << (base + G.idx(G.inv(u))))
            | (1 << (base + G.idx(G.inv(v)))))


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _component_probe(rows, n, rng):
    """Exact search through weight 4, then heuristic kernel probes."""
    from itertools import combinations

    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2

    pairs = set()
    found4 = False
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    if found4:
        return 4

    best = n + 1
    for _ in range(4):
        order = list(range(n))
        rng.shuffle(order)
        image = [1 << order[c] for c in range(n)]
        pivots = {}
        for original in rows:
            row = 0
            bits = original
            while bits:
                bit = bits & -bits
                row ^= image[bit.bit_length() - 1]
                bits ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for f in range(n):
            if f in pivots:
                continue
            word = 1 << f
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            basis.append(word)

        for word in basis:
            best = min(best, word.bit_count())
        for a, b in combinations(basis, 2):
            best = min(best, (a ^ b).bit_count())
        for a, b, c in combinations(basis, 3):
            best = min(best, (a ^ b ^ c).bit_count())
        if best <= 5:
            break
    return best


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


from functools import lru_cache


def _d2_clean_three(rows, width):
    """Exactly exclude every nonzero kernel support of weights 1..3."""
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    for i, first in enumerate(cols):
        for second in cols[:i]:
            if first ^ second in singles:
                return False
    return True


def _d2_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _d2_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _d2_four_words(rows, width):
    """Actual constituent four-words from equal pair syndromes."""
    cols = _dic_columns(rows, width)
    pairs, found = {}, set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = pairs.get(syndrome)
            if previous is not None:
                candidate = word ^ previous
                if candidate.bit_count() == 4:
                    found.add(candidate)
                    if len(found) >= 64:
                        return tuple(sorted(found))
            else:
                pairs[syndrome] = word
    return tuple(sorted(found))


def _d2_full_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    High-degree qubits are identified by degree, not sector coordinates.
    Unequal keys certify non-isomorphism; equality is discarded
    conservatively. No constituent-isomorphism assumption is made.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _dic_columns(rows, width)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    intersections = []
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            intersections.append((i, j, overlap))
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    histograms = [Counter() for _ in rows]
    for i, j, overlap in intersections:
        common = (
            overlap,
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        histograms[i][(weights[j],) + common] += 1
        histograms[j][(weights[i],) + common] += 1
    check_profiles = tuple(sorted(
        (weights[i], tuple(sorted(histograms[i].items())))
        for i in range(m)))

    high = [j for j, col in enumerate(cols)
            if col.bit_count() >= 10]
    if not high:
        raise ValueError("incidence lift lost its high-degree class")
    position = {j: v for v, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    anchors = [Counter() for _ in high]
    edge_weights = [Counter() for _ in high]

    for row in rows:
        support = row & high_mask
        degree = support.bit_count()
        if degree not in (1, 2):
            raise ValueError("intrinsic projection is not edge/grounding")
        bit = support & -support
        u = position[bit.bit_length() - 1]
        weight = row.bit_count()
        if degree == 1:
            anchors[u][weight] += 1
        else:
            v = position[(support ^ bit).bit_length() - 1]
            adjacency[u][v] += 1
            adjacency[v][u] += 1
            edge_weights[u][weight] += 1
            edge_weights[v][weight] += 1

    labels = [
        (cols[j].bit_count(),
         tuple(sorted(anchors[v].items())),
         tuple(sorted(edge_weights[v].items())))
        for v, j in enumerate(high)]
    neighbors = [tuple(sorted(a.items())) for a in adjacency]
    size = len(high)
    rooted = []
    for root in range(size):
        vector = [0] * size
        vector[root] = 1
        diagonal, features = [], [[] for _ in high]
        for _ in range(10):
            following = [0] * size
            for u, count in enumerate(vector):
                if count:
                    for v, multiplicity in neighbors[u]:
                        following[v] += count * multiplicity
            vector = following
            diagonal.append(vector[root])
            for v, count in enumerate(vector):
                features[v].append(count)
        rooted.append((
            labels[root], tuple(diagonal),
            tuple(sorted((labels[v], tuple(features[v]))
                         for v in range(size)))))
    intrinsic = (
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        check_profiles, tuple(sorted(rooted)))
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _d2_portfolio(G, count):
    """Immutable exhaustive incidence portfolio.

    Memoization only saves reconstruction cost. Publication returns
    copies, so slot order and caller mutation cannot change the designs.
    """
    from itertools import combinations

    if G.q != 8:
        raise ValueError("new dicyclic incidence bed requires order eight")
    q, e, width = G.q, G.e, 7 * G.q
    cap = min(9, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[]]]

    directions = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if _dic_order(G, g) == 4
    })
    if len(directions) != 3:
        raise ValueError("Q8 must have three non-involutory inverse classes")

    def regular(upper, lower, labels):
        return [
            [[e, G.elements[d]] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[G.elements[t]] for t in labels]
            + [[e, G.elements[d]] for d in lower],
        ]

    def grounded(labels):
        return [
            [[e] for _ in labels] + [[e]],
            [[G.elements[t]] for t in labels] + [[]],
        ]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    bank, fillings = [], set()

    def add_component(card, grid):
        counts = [[len(cell) for cell in row] for row in grid]
        if not _no_dead_lines(counts, counts):
            return
        if any(sum(row) > 7 for row in counts):
            return
        if any(sum(row[j] for row in counts) > 2 for j in range(7)):
            return
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
        if filling in fillings:
            return
        fillings.add(filling)

        # Both matrices come from the scaffold, including its distinct
        # left/right conventions. The zero dummy is never published.
        left, _, _ = lifted_matrices(G, grid, dummy)
        left = [row & mask for row in left]
        if not _d2_clean_three(left, width):
            return
        _, right, _ = lifted_matrices(G, dummy, grid)
        right = [row & mask for row in right]
        if not _d2_clean_three(right, width):
            return
        lw = _d2_four_words(left, width)
        rw = _d2_four_words(right, width)
        if not lw or not rw:
            return
        bank.append((card, freeze(grid), lw, rw,
                     gf2_rank(left), gf2_rank(right)))

    # Internal edges occur in balanced pairs. Distinct inverse classes
    # and matching voltages remove immediate parallel edges; actual
    # constituent screens, not order heuristics, reject triangles.
    for p in range(4):
        choices = list(combinations(directions, p))
        cross = 7 - 2 * p
        for upper in choices:
            for lower in choices:
                for labels in combinations(range(q), cross):
                    card = (0, upper, lower, labels)
                    add_component(card, regular(upper, lower, labels))

    # Grounding one layer changes rank without placing two singleton
    # columns on the same lifted check orbit.
    for labels in combinations(range(q), 6):
        add_component((1, (), (), labels), grounded(labels))

    bank.sort(key=lambda item: item[0])
    if not bank:
        raise ValueError("no kernel-three-clean Q8 incidence constituents")

    plans = sorted(
        (bank[a][0][0] + bank[b][0][0], a, b)
        for a in range(len(bank))
        for b in range(a, len(bank)))
    graph_keys, results = set(), []

    def four_witness(a, b, hx, hz):
        """Embed actual constituent words, then verify nontriviality."""
        bx, bz = _d2_basis(hx), _d2_basis(hz)
        for word in b[3]:
            for block in range(7):
                candidate = word << (block * width)
                if (_d2_outside(candidate, bx)
                        and all((row & candidate).bit_count() % 2 == 0
                                for row in hz)):
                    return "X", candidate
        for word in a[2]:
            for block in range(7):
                source, candidate = word, 0
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    candidate |= 1 << ((j * 7 + block) * q + t)
                    source ^= bit
                if (_d2_outside(candidate, bz)
                        and all((row & candidate).bit_count() % 2 == 0
                                for row in hx)):
                    return "Z", candidate
        return None

    for _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[1], b[1]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 424 or max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Rank surplus is measured on the actual matrices. The shape's
        # lower bound alone does not place this bed in the target cell.
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue

        signature = _d2_full_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        # These gates cover BOTH sectors and every commuting support,
        # not only constituent cycles or heuristic distance readings.
        if not (_d2_clean_three(hx, n) and _d2_clean_three(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified incidence lift failed CSS")
        witness = four_witness(a, b, hx, hz)
        if witness is None:
            continue
        kind, word = witness
        if word.bit_count() != 4:
            raise ValueError("four-word certificate changed weight")
        own, opposite, rank = (
            (hx, hz, rx) if kind == "X" else (hz, hx, rz))
        if any((row & word).bit_count() % 2 for row in opposite):
            raise ValueError("certified four-word does not commute")
        if gf2_rank(own + [word]) != rank + 1:
            raise ValueError("certified four-word is a stabilizer")

        results.append((
            k, signature, A, B, kind, word,
            (a[0], b[0]), (a[4:], b[4:])))
        if len(results) >= count:
            break

    results.sort(key=lambda item: (-item[0], item[1]))
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of a jointly separated certified design."""
    portfolio = _d2_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(portfolio)):
        raise ValueError("target-rate certified-four incidence catalogue exhausted")
    k, signature, A, B, kind, word, cards, ranks = portfolio[idx]
    return {
        "A": A, "B": B, "k": k,
        "distance_certificate": 4,
        "logical_type": kind, "witness": word,
        "design_cards": cards, "constituent_ranks": ranks,
        "intrinsic_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, already screened cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct Dic7 relative-order cards supply twenty inequivalent codes: the measured sixty-slot parent retained thirteen distinct Tanner graphs, and all five finalists still had distance three.
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
