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
CONCEPT = ("Cyclic Z11 = metacyclic Z11:Z1, order 11, A=B=(2,7): stratifying exact constituent four-word multiplicities before coupling should supply more novel distance-four codes than repeatedly selecting one triangle-free profile; five fresh k=287 distance-four readings would score 7.8765.")
GROUP = ("Keep the frozen group and order unchanged. Order eleven is the largest lift allowed by size factor fifty-three. This bed measured distance-four codes, but the immediate parent retained only four classes across sixty slots. Improve structural coverage and archive avoidance rather than claiming a new group-distance advantage.")
SHAPES = ("Keep equal 77-variable, 22-check constituents, n=583, q_max=11, kb=275 and guaranteed rate 25/53=0.471698. Actual verified ranks retain [0.475,0.525), preferring k=287 over k=280. The immediate parent's measured minimum logicals occupied sector-1 Acol/Bcol blocks broadly, not sector two.")
FILLING = ("Enumerate independently stepped regular cards and two singly grounded families. Exact constituent four-word multiplicities stratify pair proposals instead of concentrating all slots on their minimum. Both LP budgets remain at most nine. Root-individualized, edge-weighted high-degree contraction invariants supplement complete check intersections and exclude shown archived references. Every published code passes actual ranks, CSS and a full low-weight kernel screen, and carries a verified non-stabilizer witness: distance-four designs precede distance-three fallbacks. An immutable catalogue never pads exhausted slots with relabelings. Novelty against the unshown archive and sufficient catalogue yield remain measurement hypotheses.")
FAMILY_NAME = "r050_m11r1s1_27x27_fourword_stratified_rooted_catalog_v2"

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
def _z11_stratified_portfolio(G, count):
    """Immutable deterministic catalogue; caching changes only build cost."""
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, e = G.q, G.e
    if q != 11:
        raise ValueError("stratified catalogue requires the frozen Z11")
    cap = min(9, W_MAX)
    width = 7*q

    def x(t):
        return G.elements[t % q]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def columns(rows, n):
        out = [0]*n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def clear_two(rows, n):
        cols = columns(rows, n)
        return 0 not in cols and len(set(cols)) == n

    def label_key(values):
        return min(tuple(sorted((v-a) % q for v in values))
                   for a in values)

    def image(card, u):
        kind, upper, lower, values = card
        a = tuple(sorted(min(u*d % q, -u*d % q) for d in upper))
        b = tuple(sorted(min(u*d % q, -u*d % q) for d in lower))
        return kind, a, b, label_key(tuple(u*v % q for v in values))

    def normalize(card):
        kind, a, b, values = card
        forward = kind, a, b, label_key(values)
        if kind:
            return forward
        reverse = kind, b, a, label_key(tuple(-v % q for v in values))
        return min(forward, reverse)

    def canonical(card):
        return min(normalize(image(card, u)) for u in range(1, q))

    def regular(upper, lower, values):
        return [
            [[e, x(d)] for d in upper]
            + [[e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[e, x(d)] for d in lower],
        ]

    def build(card):
        kind, upper, lower, values = card
        if kind == 0:
            grid = regular(upper, lower, values)
        elif kind == 1:
            grid = [[[e] for _ in values] + [[e]],
                    [[x(v)] for v in values] + [[]]]
        else:
            grid = [[[e] for _ in values] + [[], [e]],
                    [[x(v)] for v in values]
                    + [[e, x(lower[0])], []]]
        assert len(grid) == 2 and all(len(row) == 7 for row in grid)
        assert all(len(set(cell)) == len(cell) <= 2
                   for row in grid for cell in row)
        return grid

    def component_info(grid):
        # Both actions are exposed by the verified scaffold helper.
        left, _, _ = lifted_matrices(G, grid, [[[]]])
        _, right, _ = lifted_matrices(G, [[[]]], grid)
        mask = (1 << width)-1
        left = [r & mask for r in left]
        right = [r & mask for r in right]
        if not (clear_two(left, width) and clear_two(right, width)):
            return None
        clean = (_no_short_kernel(left, width)
                 and _no_short_kernel(right, width))
        counts = Counter()
        cols = columns(left, width)
        collisions = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                collisions += counts[syndrome]
                counts[syndrome] += 1
        # Distinct columns ensure colliding pairs have disjoint supports.
        assert collisions % 3 == 0
        return (clean, collisions//3,
                tuple(sorted(Counter(counts.values()).items())),
                gf2_rank(left), gf2_rank(right))

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the full uncoloured Tanner graph.

        High qubit degree identifies a contraction without block labels.
        Check weights color its edges; singleton checks become anchors.
        Root individualization breaks the regular graph's initial symmetry.
        Equal keys are inconclusive and are conservatively rejected.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, n)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        if len(high) != 4*q:
            raise ValueError("unexpected high-degree sector")
        position = {j: i for i, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adjacency = [dict() for _ in high]
        anchors = [Counter() for _ in high]
        incident = [Counter() for _ in high]

        for row in rows:
            bits = row & mask
            endpoints = []
            while bits:
                bit = bits & -bits
                endpoints.append(position[bit.bit_length()-1])
                bits ^= bit
            w = row.bit_count()
            if len(endpoints) == 1:
                anchors[endpoints[0]][w] += 1
                incident[endpoints[0]][w] += 1
            elif len(endpoints) == 2:
                u, v = endpoints
                for a, b in ((u, v), (v, u)):
                    adjacency[a].setdefault(b, Counter())[w] += 1
                    incident[a][w] += 1
            else:
                raise ValueError("contraction is not an edge or anchor")

        adjacency = [
            {v: tuple(sorted(weights.items())) for v, weights in row.items()}
            for row in adjacency]
        labels = [
            (cols[j].bit_count(), tuple(sorted(anchors[u].items())),
             tuple(sorted(incident[u].items())))
            for u, j in enumerate(high)]
        size = len(high)

        # Verify translation symmetry on the extracted graph before using
        # four root representatives. Otherwise inspect all forty-four.
        symmetric = all(
            labels[i*q+h] == labels[i*q]
            and all(adjacency[i*q+h].get(j*q+t, ())
                    == adjacency[i*q].get(j*q+(t-h) % q, ())
                    for j in range(4) for t in range(q))
            for i in range(4) for h in range(q))
        roots = range(0, size, q) if symmetric else range(size)
        rooted = []
        for root in roots:
            features = [(labels[u], int(u == root)) for u in range(size)]
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            colors = [palette[f] for f in features]
            history = [tuple(sorted(Counter(features).items()))]
            for _ in range(10):
                features = [
                    (colors[u], tuple(sorted(
                        (edge, colors[v])
                        for v, edge in adjacency[u].items())))
                    for u in range(size)]
                history.append(tuple(sorted(Counter(features).items())))
                palette = {f: i for i, f in enumerate(sorted(set(features)))}
                following = [palette[f] for f in features]
                if following == colors:
                    break
                colors = following
            rooted.extend([tuple(history)]*(q if symmetric else 1))

        invariant = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(hx, hz), tuple(sorted(rooted)))
        return sha256(repr(invariant).encode("ascii")).digest()

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                if p not in pivots:
                    pivots[p] = word
                    break
                word ^= pivots[p]
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length()-1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    def witness(checks, stabilizers, n, weight):
        cols = columns(checks, n)
        own = span(stabilizers)
        if weight == 3:
            singles = {c: i for i, c in enumerate(cols)}
            for i, first in enumerate(cols):
                for j in range(i):
                    h = singles.get(first ^ cols[j])
                    if h is None or h in (i, j):
                        continue
                    word = (1 << i) | (1 << j) | (1 << h)
                    if outside(word, own):
                        return word
            return None
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                old = pairs.get(syndrome)
                if old is None:
                    pairs[syndrome] = word
                else:
                    difference = old ^ word
                    if difference.bit_count() == 4 and outside(difference, own):
                        return difference
        return None

    descriptors = set()
    for p in range(4):
        options = list(combinations(range(1, 6), p))
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                for tail in combinations(range(1, q), 6-2*p):
                    descriptors.add(canonical(
                        (0, upper, lower, (0,) + tail)))
    for tail in combinations(range(1, q), 5):
        descriptors.add(canonical((1, (), (), (0,) + tail)))
    for d in range(1, 6):
        for tail in combinations(range(1, q), 4):
            descriptors.add(canonical((2, (), (d,), (0,) + tail)))

    clean_regular, clean_grounded, broad_regular = [], [], []
    broad_profiles = set()
    for card in sorted(descriptors):
        grid = build(card)
        info = component_info(grid)
        if info is None:
            continue
        record = (card, info, freeze(grid))
        if info[0]:
            (clean_regular if card[0] == 0 else clean_grounded).append(record)
        elif card[0] == 0:
            # Broaden support geometry only after clean designs. Avoid
            # spending the fallback bank entirely on one triangle profile.
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
            profile = (info[1:3], _tanner_signature(rows, []))
            if profile not in broad_profiles:
                broad_profiles.add(profile)
                broad_regular.append(record)

    # Exclusion references only; none are published as candidates.
    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    two = regular((1, 3), (1, 3), (0, 2, 4))
    other = regular((1, 3), (1, 3), (0, 2, 6))
    one = ring(1, 1, (0, 2, 4, 6, 8))
    grounded = build((2, (), (1,), (0, 2, 4, 6, 8)))
    shown = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (ring(6, 7, (0, 1, 4, 6, 8)),
         ring(4, 4, (0, 3, 6, 8, 9))),
        (ring(1, 7, (3, 4, 5, 6, 8)),
         ring(4, 6, (0, 1, 3, 6, 10))),
        (matching(tuple(range(7))), two), (one, two), (two, two),
        (one, other),
        (two, regular((2, 5), (2, 5), (0, 3, 7))),
        (two, regular((2, 3), (2, 3), (0, 1, 5))),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
        (two, grounded), (matching(tuple(range(7))), grounded),
        (matching(tuple(range(7))),
         build((1, (), (), tuple(range(6))))),
        (other, regular((1, 3), (1, 3), (0, 2, 7))),
        (two, regular((1, 4), (1, 4), (0, 2, 8))),
        (matching((0, 1, 2, 3, 4, 6, 7)),
         matching((0, 1, 2, 4, 5, 6, 8))),
        (one, build((1, (), (), (0, 1, 3, 4, 7, 9)))),
        (one, regular((1, 3), (1, 3), (0, 2, 6))),
        (one, regular((1, 3), (1, 3), (0, 2, 4))),
        (one, regular((1, 2), (2, 5), (0, 3, 7))),
        (two, regular((1, 3), (1, 4), (0, 2, 4))),
    ]
    graph_keys = set()
    for A, B in shown:
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(graph_key(hx, hz, n))

    def stratified_plans(first, second, same, seed):
        buckets = {}
        for ai, a in enumerate(first):
            for bi, b in enumerate(second):
                if same and bi < ai:
                    continue
                # Exact four-word counts define deliberate structural
                # coverage strata, not a quantum-distance prediction.
                key = (
                    tuple(sorted((a[1][1], b[1][1]))),
                    tuple(sorted((len(a[0][1]), len(b[0][1])))))
                bucket = buckets.setdefault(key, [])
                bucket.extend((ai, bi, u) for u in range(1, q))
        local = random.Random(BASE_SEED ^ seed)
        for key in sorted(buckets):
            local.shuffle(buckets[key])
        keys = sorted(buckets)
        for depth in range(max((len(v) for v in buckets.values()), default=0)):
            for key in keys:
                if depth < len(buckets[key]):
                    yield buckets[key][depth]

    phases = [
        (clean_regular, clean_regular, True, 4, 0x114041),
        (clean_regular, clean_grounded, False, 4, 0x114042),
        (clean_regular + broad_regular, broad_regular, False, 3, 0x114043),
    ]
    results, filling_keys = [], set()
    for first, second, same, floor, seed in phases:
        examined = 0
        for ai, bi, scale in stratified_plans(first, second, same, seed):
            a, b = first[ai], second[bi]
            A, B = a[2], build(image(b[0], scale))
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            examined += 1
            if examined > 12000:
                break
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 583 or not (475*n <= 1000*k < 525*n):
                continue
            if max(r.bit_count() for r in hx + hz) > cap:
                continue
            if not (clear_two(hx, n) and clear_two(hz, n)):
                continue
            if floor == 4 and not (
                    _no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
                continue

            signature = graph_key(hx, hz, n)
            if signature in graph_keys:
                continue
            word = witness(hz, hx, n, floor)
            kind = "X"
            if word is None:
                word = witness(hx, hz, n, floor)
                kind = "Z"
            if word is None:
                continue
            checks = hz if kind == "X" else hx
            assert word.bit_count() == floor
            assert all((word & row).bit_count() % 2 == 0 for row in checks)
            if not css_ok(hx, hz):
                raise ValueError("verified stratified lift failed CSS")
            graph_keys.add(signature)
            results.append((
                k*floor*floor/n, freeze(A), freeze(B), k, floor,
                kind, word, (a[0], image(b[0], scale)),
                (a[1][1], b[1][1]), signature.hex()))
            if len(results) >= count:
                return tuple(sorted(results, key=lambda r: r[0], reverse=True))

    # Never manufacture additional slots from an exhausted equivalence class.
    return tuple(sorted(results, key=lambda r: r[0], reverse=True))


def NEW_CTX(rng, G, idx):
    """Pure indexed selection from an immutable, jointly screened portfolio."""
    catalogue = _z11_stratified_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(catalogue)):
        raise ValueError("novel stratified Z11 catalogue exhausted")
    quality, A, B, k, distance, kind, word, cards, counts, signature = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "certified_distance": distance,
        "verified_logical_type": kind, "verified_logical": word,
        "constituent_four_counts": counts, "design_cards": cards,
        "intrinsic_signature": signature, "certified_score": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete screened cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty evaluator-distinct codes: the immediate parent's sixty slots retained only four classes, measured at distances 4,4,3,3.
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
