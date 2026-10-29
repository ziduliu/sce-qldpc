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
CONCEPT = ("Cyclic Z11 = metacyclic Z11:Z1, order 11, A=B=(2,7): jointly selecting independently stepped regular constituents should supply five fresh k=287 distance-four finalists, raising fitness toward 287*16/583=7.8765; novelty and measurement survival remain falsifiable.")
GROUP = ("Keep the frozen group and order byte-for-byte unchanged. This bed measured distance-four codes, and q=11 is the largest admissible lift for size factor 53. The parent retained five classes from sixty slots, including two lower-rank k=280 finalists; improve joint catalogue coverage rather than claim a group-distance advantage.")
SHAPES = ("Keep equal 77-variable, 22-check constituents, n=583, q_max=11, kb=275 and guaranteed rate 25/53=0.47170. Complete verified ranks must give k=287, rate 0.49228, in cell 0.500. Parent minimum logicals broadly occupied sector-1 Acol/Bcol blocks, so vary both constituents and their relative orientations.")
FILLING = ("Enumerate triangle-free regular incidence cards with independent upper/lower cycle steps and seven total columns. Every component row has seven terms and every column two, giving BOTH CSS maxima nine. Build one immutable deterministic joint catalogue, not separate slot ownership bins. Verified full ranks, CSS and complete kernel exclusion through three gate publication. Exact stabilizer-aware weight-four counts prioritize candidates without mistaking multiplicity for distance. Root-individualized intrinsic contraction invariants and optional canonical full-Tanner certificates reject repeats and shown archive references. Unknown archive coverage and five-fresh-finalist yield remain evaluator-tested hypotheses.")
FAMILY_NAME = "r050_m11r1s1_27x27_joint_regular_voltage_fourcount_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same metacyclic construction rule; factor 53 requires q <= 11.
    # Z_11 x| Z_1, with s^r = 1 mod m.
    m, r, s = 11, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _component_spec(rng, mode, q):
    """Matching columns, or two cycle columns plus five matchings."""
    if mode == 0:
        return (), tuple(sorted(rng.sample(range(q), 7)))
    if mode == 1:
        d = rng.randrange(1, q)
        phase = rng.randrange(q)
        # Five independent vertices of the eleven-cycle with step d.
        labels = tuple(sorted((phase + d * t) % q
                              for t in (0, 2, 4, 6, 8)))
        return (d, d), labels
    return (rng.randrange(1, q), rng.randrange(1, q)), \
        tuple(sorted(rng.sample(range(q), 5)))


def _component_cells(G, spec):
    ds, labels = spec
    e, els = G.e, G.elements
    if not ds:
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]
    # Column weights remain two; both row weights remain seven.
    return [[[e, els[ds[0]]]] + [[e] for _ in labels] + [[]],
            [[]] + [[els[s]] for s in labels] + [[e, els[ds[1]]]]]


def _translation_key(labels, q):
    return min(tuple(sorted((s + t) % q for s in labels))
               for t in range(q))


def _scaled_spec_key(spec, u, q):
    ds, labels = spec
    s = tuple((u * x) % q for x in labels)
    forward = _translation_key(s, q)
    backward = _translation_key(tuple((-x) % q for x in s), q)
    if not ds:
        return (0, min(forward, backward))
    d0, d1 = ((u * d) % q for d in ds)
    d0, d1 = min(d0, q - d0), min(d1, q - d1)
    return min((1, d0, d1, forward),
               (1, d1, d0, backward))


def _pair_key(a, b, q):
    """Remove translations, row swaps, common automorphisms and A/B swap."""
    keys = []
    for u in range(1, q):
        ka = _scaled_spec_key(a, u, q)
        kb = _scaled_spec_key(b, u, q)
        keys.append(tuple(sorted((ka, kb))))
    return min(keys)


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


def _no_short_kernel(rows, n):
    """Exactly exclude kernel vectors of weights one, two and three."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    seen = set(columns)
    if 0 in seen or len(seen) != n:
        return False
    for i, c in enumerate(columns):
        for j in range(i):
            if (c ^ columns[j]) in seen:
                return False
    return True


from functools import lru_cache


@lru_cache(maxsize=1)
def _joint_voltage_portfolio(G, count):
    """Pure deterministic joint selection; memoization changes only cost.

    Published grids are immutable. Every catalogue entry passes the
    complete verified-matrix screens, and no slot is padded with copies.
    """
    from itertools import combinations

    q, e, els = G.q, G.e, G.elements
    cap = min(9, W_MAX)

    def label_key(labels):
        # Translation is a component gauge. Reversal is handled together
        # with the row swap, preserving independent layer orientations.
        return min(tuple(sorted((s - anchor) % q for s in labels))
                   for anchor in labels)

    def regular(upper, lower, labels):
        """Seven degree-two columns; row totals seven."""
        p = len(upper)
        assert len(lower) == p and len(labels) + 2 * p == 7
        return (
            [[e, els[d]] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[els[s]] for s in labels]
            + [[e, els[d]] for d in lower],
        )

    def normalized_card(p, upper, lower, labels):
        forward = (p, tuple(upper), tuple(lower), label_key(labels))
        reverse = (p, tuple(lower), tuple(upper),
                   label_key(tuple(-s % q for s in labels)))
        return min(forward, reverse)

    def card_cells(card):
        _, upper, lower, labels = card
        return [list(row) for row in regular(upper, lower, labels)]

    # Exhaustive structural prescriptions, including unequal upper and
    # lower step sets. Do not quotient independent group automorphisms:
    # relative orientations can change the complete coupled code.
    by_profile = []
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q + 1) // 2), p):
            signed = set(steps) | {(-d) % q for d in steps}
            if not any((a + b) % q in signed
                       for a in signed for b in signed):
                options.append((steps, signed))
        cards = set()
        for upper, su in options:
            for lower, sl in options:
                forbidden = su | sl
                for tail in combinations(range(1, q), 6 - 2 * p):
                    labels = (0,) + tail
                    if any((a - b) % q in forbidden
                           for a, b in combinations(labels, 2)):
                        continue
                    cards.add(normalized_card(
                        p, upper, lower, labels))
        screened = []
        for card in sorted(cards):
            grid = card_cells(card)
            left, _, _ = lifted_matrices(G, grid, [[[]]])
            width = 7 * q
            mask = (1 << width) - 1
            rows = [row & mask for row in left]
            if (gf2_rank(rows) == 2 * q - 1
                    and _no_short_kernel(rows, width)):
                screened.append(card)
        by_profile.append(screened)

    # Interleave support profiles instead of spending an initial
    # prefix entirely on one regular matching model.
    cards = []
    for turn in range(max(map(len, by_profile))):
        for profile in (2, 1, 0):
            if turn < len(by_profile[profile]):
                cards.append(by_profile[profile][turn])

    def columns(rows, n):
        result = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def fingerprint(hx, hz, n):
        """Intrinsic full-Tanner key, with optional canonical certification.

        The fallback combines complete check intersections with rooted
        refinement of the intrinsically identified high-degree sector.
        Equal keys are discarded conservatively; equality is not itself
        asserted to prove isomorphism.
        """
        from collections import Counter
        from hashlib import sha256

        rows = list(hx) + list(hz)
        cols = columns(rows, n)
        try:
            import pynauty
        except ImportError:
            pynauty = None

        if pynauty is not None:
            m = len(rows)
            adjacency = {v: [] for v in range(m + n)}
            for r, row in enumerate(rows):
                bits = row
                while bits:
                    bit = bits & -bits
                    v = m + bit.bit_length() - 1
                    adjacency[r].append(v)
                    adjacency[v].append(r)
                    bits ^= bit
            classes = {}
            for v, neighbors in adjacency.items():
                classes.setdefault(len(neighbors), set()).add(v)
            graph = pynauty.Graph(
                number_of_vertices=m + n, directed=False,
                adjacency_dict=adjacency,
                vertex_coloring=[classes[d] for d in sorted(classes)])
            certificate = pynauty.certificate(graph)
            return sha256(b"canonical-tanner:" + certificate).digest()

        high = [j for j, col in enumerate(cols)
                if col.bit_count() >= 10]
        positions = {j: v for v, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        anchors = [Counter() for _ in high]
        for row in rows:
            bits = row & mask
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("incidence contraction lost its endpoints")
            first = bits & -bits
            u = positions[first.bit_length() - 1]
            weight = row.bit_count()
            if degree == 1:
                anchors[u][weight] += 1
            else:
                v = positions[(bits ^ first).bit_length() - 1]
                adjacency[u][(v, weight)] += 1
                adjacency[v][(u, weight)] += 1

        labels = [
            (cols[j].bit_count(), tuple(sorted(anchors[v].items())))
            for v, j in enumerate(high)]

        def recolor(features):
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        rooted = []
        for root in range(len(high)):
            colors = recolor([
                (labels[v], int(v == root)) for v in range(len(high))])
            history = []
            for _ in range(10):
                features = [
                    (colors[v], tuple(sorted(
                        (weight, colors[w], multiplicity)
                        for (w, weight), multiplicity
                        in adjacency[v].items())))
                    for v in range(len(high))]
                following = recolor(features)
                history.append(tuple(sorted(Counter(features).items())))
                old_classes = len(set(colors))
                colors = following
                if len(set(colors)) == old_classes:
                    break
            rooted.append((colors[root], tuple(history)))

        intrinsic = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(list(hx), list(hz)),
            tuple(sorted(rooted)),
        )
        return sha256(repr(intrinsic).encode("ascii")).digest()

    # Explicitly exclude the shown archived Z11 codes. These matrices
    # are only fingerprint references, never submitted as candidates.
    def old_cycle(d0, d1, labels):
        return [list(row) for row in regular((d0,), (d1,), labels)]

    def old_matching(labels):
        return [list(row) for row in regular((), (), labels)]

    archived = [
        (old_matching((0, 1, 2, 4, 5, 6, 10)),
         old_matching((0, 2, 3, 5, 6, 7, 9))),
        (old_cycle(10, 10, (0, 2, 5, 7, 9)),
         old_cycle(10, 10, (0, 2, 4, 7, 9))),
        (old_cycle(6, 7, (0, 1, 4, 6, 8)),
         old_cycle(4, 4, (0, 3, 6, 8, 9))),
        (old_cycle(1, 7, (3, 4, 5, 6, 8)),
         old_cycle(4, 6, (0, 1, 3, 6, 10))),
        (old_matching(tuple(range(7))),
         [list(row) for row in
          regular((1, 3), (1, 3), (0, 2, 4))]),
        (old_matching(tuple(range(7))),
         [[[e] for _ in range(7)],
          [[els[s]] for s in range(6)] + [[]]]),
    ]
    # Additional regular representatives shown in the current table and
    # related programmes. References are exclusions, never output seeds.
    two = [list(row) for row in
           regular((1, 3), (1, 3), (0, 2, 4))]
    other_two = [list(row) for row in
                 regular((1, 3), (1, 3), (0, 2, 6))]
    archived.extend([
        (two, two),
        (old_cycle(1, 1, (0, 2, 4, 6, 8)), other_two),
        (old_matching(tuple(range(7))),
         [list(row) for row in
          regular((1, 4), (1, 4), (0, 2, 5))]),
        (other_two,
         [list(row) for row in
          regular((1, 3), (1, 3), (0, 2, 7))]),
        (two,
         [list(row) for row in
          regular((1, 4), (1, 4), (0, 2, 8))]),
        (old_matching((0, 1, 2, 3, 4, 6, 7)),
         old_matching((0, 1, 2, 4, 5, 6, 8))),
    ])
    archive_keys = set()
    for A, B in archived:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    def scaled_card(card, u):
        p, upper, lower, labels = card
        a = tuple(sorted(min((u * d) % q, (-u * d) % q)
                         for d in upper))
        b = tuple(sorted(min((u * d) % q, (-u * d) % q)
                         for d in lower))
        return normalized_card(
            p, a, b, tuple((u * s) % q for s in labels))

    def regular_pair_key(a, b):
        # Common automorphisms, column gauges, row swaps and A/B swap.
        return min(tuple(sorted((scaled_card(a, u),
                                 scaled_card(b, u))))
                   for u in range(1, q))

    def logical_four_count(checks, stabilizers, n):
        """Exactly count non-stabilizer kernel supports of weight four.

        Called only after complete kernel exclusion through three.
        Equal pair syndromes then have disjoint supports. Their residues
        modulo the stabilizer span decide whether the resulting word is
        logical. Each four-support has three unordered pair partitions.
        """
        pivots = {}
        for word in stabilizers:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        ordered = sorted(pivots, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in ordered:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residues.append(word)

        cols = columns(checks, n)
        pairs = {}
        collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                residue = residues[i] ^ residues[j]
                record = pairs.get(syndrome)
                if record is None:
                    pairs[syndrome] = [1, {residue: 1}]
                else:
                    total, counts = record
                    same = counts.get(residue, 0)
                    collisions += total - same
                    record[0] = total + 1
                    counts[residue] = same + 1
        if collisions % 3:
            raise ValueError("logical-four partition count is inconsistent")
        return collisions // 3

    # Jointly cover relative orientations. A common automorphism is
    # removed by regular_pair_key, but independent scalings are trials,
    # not automatically emitted as distinct designs.
    plans = [
        (a, scaled_card(b, u))
        for ai, a in enumerate(cards)
        for b in cards[ai:]
        for u in range(1, q)
    ]
    local = random.Random(BASE_SEED ^ 0x11F42704)
    local.shuffle(plans)
    plans.sort(key=lambda pair: (
        int(pair[0][1] == pair[0][2])
        + int(pair[1][1] == pair[1][2]),
        -pair[0][0] - pair[1][0],
        pair[0] == pair[1],
    ))

    graph_keys, orbit_keys = set(), set()
    entries = []
    examined = 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for a, b in plans:
        orbit = regular_pair_key(a, b)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)
        examined += 1
        if examined > 2048:
            break

        A, B = card_cells(a), card_cells(b)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 287 or not (475 * n <= 1000 * k < 525 * n):
            continue

        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified joint regular lift failed CSS")

        nx = logical_four_count(hz, hx, n)
        nz = logical_four_count(hx, hz, n)
        # A positive count supplies a weight-four logical. If both
        # counts vanish, the exact screens instead establish floor five.
        floor = 5 if nx == nz == 0 else 4
        quality = (k * floor * floor / n, k,
                   -max(nx, nz), -nx - nz, signature)
        graph_keys.add(signature)
        entries.append((
            quality, freeze(A), freeze(B), k, floor, (nx, nz), (a, b)))
        if len(entries) >= count + 24:
            break

    entries.sort(key=lambda entry: entry[0], reverse=True)
    # Never pad an exhausted catalogue with relabelings.
    return tuple(entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable jointly screened design."""
    entries = _joint_voltage_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("novel joint regular catalogue exhausted")
    quality, A, B, k, floor, fours, cards = entries[idx]
    return {
        "A": A, "B": B, "k": k,
        "verified_logical_floor": floor,
        "exact_logical_four_counts": fours,
        "design_cards": cards,
        "selection_quality": quality[:4],
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without changing the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: Fixed opposite-cycle grounding plus symmetric-layer regular cards supplies twenty inequivalent designs: the current sixty-slot evaluation retained five classes, all measured at distance four, with two k=280 and three k=287 finalists.
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
