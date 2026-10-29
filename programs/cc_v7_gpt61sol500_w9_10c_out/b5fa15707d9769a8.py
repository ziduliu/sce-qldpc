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
CONCEPT = ("C3 semidirect C4 with action 2, order 12, A=(2,6), B=(2,7): rank-prioritized balanced, asymmetric and mixed-degree constituents should supply five novel target-band codes improving fitness above 7.4780; exact logical exclusion through four tests the stronger distance-five hypothesis.")
GROUP = ("Keep the cyclic-by-cyclic semidirect-product construction rule. Factor 46 excludes order 24 and permits q<=13; m=3,r=4,s=2 gives order twelve and 2^4=1 mod 3. Related order-12 (2,6)x(2,7) programs measured k=258,d30s=4, but also substantial duplicate collapse. Their readings motivate support diversification, not a transferred distance guarantee.")
SHAPES = ("A=(2,6), B=(2,7) gives n=552, kb=240 and guaranteed rate 20/46=10/23 in cell 0.450. Components have 72 variables/24 checks and 84 variables/24 checks. Relative to the parent, A gains a check row, B's component length halves, and sector 2 has four block pairs. Parent measured minima occupied sector-1 Acol0..2/Bcol0..6 broadly. Verified complete ranks enforce [0.425,0.475).")
FILLING = ("Optimize twenty structural prescriptions crossing five A incidence allocations with four B allocations; later proposals add singleton/degree-three columns and other count profiles. Regular A rows total six/six or five/seven; B rows total seven. Degree-three B columns require balanced A. BOTH CSS sums must stay <=min(9,W_MAX). Complete stabilizer-aware tests exclude logicals below four and prioritize exclusion through four, then actual k. Immutable rooted uncoloured Tanner invariants conservatively reject repeats and shown archive anchors. Distance-five yield, unseen-archive novelty and five-code improvement remain falsifiable hypotheses.")
FAMILY_NAME = "r045_m3r4s2_26x27_asymmetric_logical4_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 46 excludes order 24; 2^4=1 modulo 3.
    m, r, s = 3, 4, 2
    if not valid_q(q):
        raise ValueError("this semidirect product has order 12")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + powers[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-powers[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _kernel_profile(rows, n):
    """Exact low-weight kernel screen, not a distance-search estimate.

    A reading of five means only that weights one through four were
    excluded. Equal syndromes of words of weight <=2 detect all such
    kernel words. Collision counts are only a secondary selection feature.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= check
            row ^= low
    seen = {0: 0}
    lightest, collisions = 5, 0

    def inspect(syndrome, word):
        nonlocal lightest, collisions
        if syndrome in seen:
            difference = word ^ seen[syndrome]
            if difference:
                lightest = min(lightest, difference.bit_count())
                collisions += 1
        else:
            seen[syndrome] = word

    for j, syndrome in enumerate(columns):
        inspect(syndrome, 1 << j)
    for j, syndrome in enumerate(columns):
        for i in range(j):
            inspect(syndrome ^ columns[i], (1 << j) | (1 << i))
    return lightest, collisions


def _extract_components(G, hx, hz):
    """Both constituent matrices, extracted from the verified full lift."""
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
    br = [row & ((1 << (cb * q)) - 1)
          for row in hz[:rb * q]]
    return ar, br


def _shape_draw(rng, G, attempt):
    """Optimize twenty support prescriptions, not twenty label draws."""
    card, epoch = attempt % 20, attempt // 20
    ap, bp = divmod(card, 4)
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })
    outside = [g for g in directions if g[1] % 2]
    inside = [g for g in directions if not g[1] % 2]
    all_labels = list(G.elements)

    def incidence(width, internal, policy):
        if policy == 0:
            steps = outside
            labels = [g for g in G.elements
                      if g[1] % 2 == epoch % 2]
        elif policy == 1:
            steps = inside
            labels = [g for g in G.elements if g[1] % 2 == 0]
        elif policy == 2:
            steps, labels = directions, all_labels
        elif policy == 3:
            steps = outside
            labels = [g for g in G.elements if g[1] in (0, 1)]
        else:
            steps = directions
            labels = [g for g in G.elements
                      if g[1] % 2 == (epoch // 5) % 2]

        if any(count > len(steps) for count in internal):
            return None
        C = [[], []]
        for i, count in enumerate(internal):
            for gap in rng.sample(steps, count):
                C[i].append([G.e, gap])
                C[1 - i].append([])

        cross = width - sum(internal)
        if not sum(internal):
            labels = all_labels
        if cross < 0 or cross > len(labels):
            return None
        for voltage in rng.sample(labels, cross):
            C[0].append([G.e])
            C[1].append([voltage])
        return C

    # Last two prescriptions have row totals five/seven, rather than
    # merely permuting one balanced six/six constituent.
    allocations = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 1))
    A = incidence(6, allocations[ap], (epoch + ap) % 5)
    B = incidence(7, (bp, bp), (epoch // 5 + bp) % 5)
    if A is None or B is None:
        return None

    def fill_profile(profile):
        C = [[], []]
        for counts in profile:
            for i, count in enumerate(counts):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(directions)
                    cell = [first, G.mul(first, gap)]
                else:
                    cell = rng.sample(all_labels, count)
                C[i].append(cell)
        return C

    if ap < 3 and bp >= 2 and epoch % 4 == 3:
        # Both B rows total seven. Column degrees one/three are genuine
        # topology changes. A rows total six, so HX stays <=6+3.
        profiles = (
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
            ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
            ((1, 0), (3, 0)) + ((1, 1),) * 3
            + ((0, 2),) * 2,
        )
        B = fill_profile(profiles[(epoch // 4 + bp) % 3])

    if attempt >= 960 and ap < 3 and epoch % 8 == 7:
        # A degree-three columns require B rows <=6. These broader
        # proposals receive the same complete logical tests; no floor
        # or rate relaxation is used if their hypothesis fails.
        A = fill_profile(
            ((1, 0), (1, 2)) + ((1, 1),) * 4)
        B = fill_profile(
            ((1, 0), (0, 1)) + ((1, 1),) * 5)

    sa = [[len(c) for c in row] for row in A]
    sb = [[len(c) for c in row] for row in B]
    if not _no_dead_lines(sa, sb):
        return None
    if _grid_weights(sa, sb) > min(9, W_MAX):
        return None
    assert len(A) == 2 and all(len(row) == 6 for row in A)
    assert len(B) == 2 and all(len(row) == 7 for row in B)
    assert all(len(set(c)) == len(c) <= 5
               for grid in (A, B) for row in grid for c in row)
    return A, B, (card, epoch % 5)


def _no_kernel_through_three(rows, n):
    """Exactly exclude all nonzero kernel words of weights 1, 2 and 3."""
    columns = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, first in enumerate(columns):
        for j in range(i):
            if (first ^ columns[j]) in singles:
                return False
    return True


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
    return tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)))


def _shape_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _shape_logical4(checks, stabilizers, n):
    """Exact logical existence through four, modulo stabilizer span.

    Five means exclusion through four, not a distance-search reading.
    Burden counts detected collisions, not distinct logical operators.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = row
                break
            row ^= previous

    def reduce(word):
        remainder = 0
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                bit = 1 << p
                remainder |= bit
                word ^= bit
            else:
                word ^= previous
        return remainder

    columns = _shape_columns(checks, n)
    units = [reduce(1 << i) for i in range(n)]
    seen = {0: (0, 0)}
    floor, burden = 5, 0

    for i, syndrome in enumerate(columns):
        word, quotient = 1 << i, units[i]
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = (word, quotient)
        elif quotient != previous[1]:
            return (word ^ previous[0]).bit_count(), 1

    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            word = (1 << i) | (1 << j)
            quotient = units[i] ^ units[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = (word, quotient)
            elif quotient != previous[1]:
                weight = (word ^ previous[0]).bit_count()
                if weight < 4:
                    return weight, burden + 1
                floor = 4
                burden += 1
    return floor, burden


def _shape_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Root high-degree qubits identified by degree, never LP labels.
    Projected check hyperedges and low-qubit corners retain endpoint
    multiplicity. Different signatures certify graph inequivalence.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _shape_columns(rows, n)
    high = [j for j, c in enumerate(columns) if c.bit_count() >= 10]
    position = {j: v for v, j in enumerate(high)}
    endpoints, tags, check_ends = [], [], []

    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in position:
                ends.append(position[j])
            bits ^= bit
        ends = tuple(sorted(ends))
        check_ends.append(ends)
        endpoints.append(ends)
        tags.append((0, row.bit_count()))

    for j, syndrome in enumerate(columns):
        if j in position:
            continue
        ends, weights = [], []
        bits = syndrome
        while bits:
            bit = bits & -bits
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            bits ^= bit
        endpoints.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(endpoints):
        for v in ends:
            incident[v].append(edge)

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(4):
            edge_features = [
                (tag, tuple(sorted(colours[v] for v in ends)))
                for tag, ends in zip(tags, endpoints)
            ]
            edge_colours = compress(edge_features)
            vertex_features = [
                (colours[v],
                 tuple(sorted(edge_colours[e] for e in incident[v])))
                for v in range(len(high))
            ]
            history.append((
                tuple(sorted(Counter(edge_features).items())),
                tuple(sorted(Counter(vertex_features).items())),
            ))
            colours = compress(vertex_features)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    return (
        n,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )


def _shape_archive_keys(G):
    """Shown order-12 anchors only; never publish these fillings."""
    def incidence(top, bottom, voltages):
        B = [[], []]
        for gap in top:
            B[0].append([G.e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([G.e, gap])
        for voltage in voltages:
            B[0].append([G.e])
            B[1].append([voltage])
        return B

    # Encoded directly in the C3:C4 presentation used by the shown
    # order-12 (2,6)x(2,7) bed. Exclude full graphs, not label packets.
    A = [
        [[G.e, g] for g in ((0, 1), (2, 1), (1, 1))]
        + [[], [], []],
        [[], [], []]
        + [[G.e, g] for g in ((1, 1), (2, 1), (0, 1))],
    ]
    cards = (
        ((), (),
         ((0,1),(3 % 3,1),(2,0),(1,0),(1,1),(0,0),(4 % 3,1))),
        (((1,1),(0,1)), ((2,1),(0,1)),
         ((0,0),(0,0),(1,0))),
        (((1,1),), ((2,1),),
         ((2,0),(0,0),(0,0),(1,0),(2,0))),
        (((2,1),(0,1),(1,1)), ((0,1),(1,1),(2,1)), ((2,0),)),
    )
    # Some shown tables use a different isomorphic group encoding.
    # Only already-valid packets in THIS encoding are references here;
    # the unseen global archive remains authoritative.
    keys = set()
    for top, bottom, voltages in cards:
        B = incidence(top, bottom, voltages)
        hx, hz, n = lifted_matrices(G, A, B)
        keys.add(_shape_signature(hx, hz, n))
    return keys


from functools import lru_cache


@lru_cache(maxsize=1)
def _shape_portfolio(G):
    """Deterministic immutable portfolio, independent of slot call order.

    Memoization saves rebuilding work; it does not change selection.
    Every screen uses the evaluator's verified lifted matrices.
    """
    local = random.Random(BASE_SEED ^ 0x12342675)
    cap = min(9, W_MAX)
    graph_keys = _shape_archive_keys(G)
    filling_keys, entries = set(), []
    target_count = max(80, N_SLOTS + 20)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(64):
        candidates = []
        for trial in range(64):
            attempt = 64 * batch + trial
            proposal = _shape_draw(local, G, attempt)
            if proposal is None:
                continue
            A, B, card = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) <= cap

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _extract_components(G, hx, hz)
            qa = _kernel_profile(ar, 6 * G.q)
            qb = _kernel_profile(br, 7 * G.q)
            # Constituent words are diagnostics, not automatically
            # logicals. The complete stabilizer-quotient screens decide.
            priority = (
                -min(qa[0], qb[0]), -k,
                qa[1] + qb[1], attempt,
            )
            candidates.append(
                (priority, A, B, hx, hz, n, k, rx, rz, card, qa, qb))

        candidates.sort(key=lambda record: record[0])
        survivors = []
        for record in candidates[:20]:
            (_, A, B, hx, hz, n, k, rx, rz, card, qa, qb) = record
            xf = _shape_logical4(hz, hx, n)
            if xf[0] < 4:
                continue
            zf = _shape_logical4(hx, hz, n)
            if zf[0] < 4:
                continue
            floor = min(xf[0], zf[0])
            quality = (
                k * floor * floor / n,
                floor, k, -(xf[1] + zf[1]),
            )
            survivors.append((quality, record, xf, zf))

        survivors.sort(key=lambda item: item[0], reverse=True)
        added = 0
        for quality, record, xf, zf in survivors:
            (_, A, B, hx, hz, n, k, rx, rz, card, qa, qb) = record
            signature = _shape_signature(hx, hz, n)
            if signature in graph_keys:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified new-shape lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(signature)
            screen = (
                ("k", k),
                ("card", card),
                ("logical_profiles", (xf, zf)),
                ("component_profiles", (qa, qb)),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("A_row_totals", tuple(map(sum, sa))
                 if False else tuple(sum(map(len, row)) for row in A)),
                ("logical_exclusion_through", min(xf[0], zf[0]) - 1),
            )
            entries.append((quality, freeze(A), freeze(B), screen))
            added += 1
            if added == 8:
                break

        if len(entries) >= target_count:
            break

    # Compare the complete optimized pool before publishing its slots.
    # Never fill shortages with recycled codes or automorphism copies.
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete screened new-shape design."""
    portfolio = _shape_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("novel target-band shape portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: The C12:C2(s=5) rank-first odd-incidence prefix supplies twenty inequivalent initial designs: slots 0..19 collapsed to four classes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
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
