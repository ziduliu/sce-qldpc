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
CONCEPT = ("On metacyclic C11 semidirect_2 C10, order 110, A=B=(1,2), independently optimized cubic constituents should supply five novel finalists with d30s>18 in cell 0.200. Actual kernel witnesses guide selection; neither probe caps nor unsuccessful searches certify that hypothesis.")
GROUP = ("Keep the parent's metacyclic construction rule, changing its parameters to m=11,r=10,s=2. Since 2 has order ten modulo eleven, this is a valid order-110 semidirect product. The frontier order-110, centre-one, eleven-class (1,2)x(1,2) bed measured five [550,110,18] finalists and fitness 64.8. The new factor five permits q<=119; q=110 uses twice the parent's lift order without changing the presentation rule.")
SHAPES = ("Change both sides to (1,2). Equal constituents have 220 variables and 110 checks, n=550, kb=110 and guaranteed rate 0.200. Full verified ranks retain [0.175,0.225). Parent weight-five minima lay entirely in sector-1 Acol/Bcol blocks across its three columns; replacing its degree-two constituents with cubic ones targets those short cycles rather than moving their labels. Earlier mixed-sector failures motivate complete-matrix screening.")
FILLING = ("All four cells contain three distinct elements: constituent row totals six and column totals three give BOTH CSS maxima nine. Structural cards vary triangle relative-order spectra (10,10,11), (5,5,11), (5,10,10), and (5,5,5). Verified constituent lifts exclude kernels through six using regular-action anchored triples, then kernel-witness probes and Tanner-cycle penalties optimize the bank. Complete candidates pass actual-rank, CSS, weight and kernel-through-four gates. Intrinsic uncoloured Tanner signatures reject equal classes and the shown frontier reference. Capped witness readings are selection heuristics, not quantum-distance certificates; the unseen archive and d30s remain evaluator-tested.")
FAMILY_NAME = "r020_metacyclic11by10_s2_12x12_cubic_spectrum_portfolio"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 110

def custom_ops(q):
    # Same metacyclic presentation rule, with new parameters.
    # x^11=y^10=1, y*x*y^-1=x^2; 2 has order ten modulo 11.
    if not valid_q(q):
        raise ValueError("this metacyclic group has order 110")
    m, r, s = 11, 10, 2
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


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


def _component_rows(G, side, grid):
    """Extract any shaped constituent using the verified lift conventions."""
    dummy = [[[G.e]]]
    rows, cols = len(grid), len(grid[0])
    mask = (1 << (cols * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx[:rows * G.q]]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz[:rows * G.q]]


def _generator_key(G, generators):
    """Quotient known component orbit copies, not arbitrary graph isos.

    Affine-group automorphisms have x->x^u, y->x^v*y. Binomial
    generators can also be inverted and their columns permuted.
    """
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    keys = []
    for u in range(1, 11):
        for v in range(11):
            images = []
            for a, b in generators:
                g = ((u * a + v * sums[b]) % 11, b)
                images.append(min(G.idx(g), G.idx(G.inv(g))))
            keys.append(tuple(sorted(images)))
    return min(keys)


def _component_bank(G):
    """Deterministically optimize different quotient prescriptions."""
    local = random.Random(BASE_SEED ^ 0x550403)
    # Quotient-coordinate profiles distinguish construction prescriptions.
    # Avoid two rotations, which commute and immediately create a 4-word.
    profiles = ((0, 1, 2), (0, 1, 1), (1, 1, 2),
                (1, 2, 2), (0, 2, 2), (1, 2, 3))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    entries, used = [], set()
    for attempt in range(256):
        profile = profiles[attempt % len(profiles)]
        generators = [local.choice(pools[b]) for b in profile]
        undirected = [min(G.idx(g), G.idx(G.inv(g)))
                      for g in generators]
        if len(set(undirected)) != 3:
            continue
        key = _generator_key(G, generators)
        if key in used:
            continue
        used.add(key)
        A = [[[G.e, g] for g in generators]]
        reading, collisions = _kernel_profile(
            _component_rows(G, "A", A), 3 * G.q)
        entries.append(((reading, -collisions), key, A))
    if not entries:
        raise ValueError("component bank unexpectedly empty")
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Prefer every available clean component. If the hypothesis fails,
    # retain the best legal components instead of invalidating all slots.
    clean = [item for item in entries if item[0][0] == 5]
    return (clean if clean else entries[:12])[:16]


def _draw_B(rng, G, attempt, bank):
    """Indexed support/order prescriptions, subsequently screened."""
    if attempt % 4 == 0:
        source = bank[(attempt // 4) % len(bank)][2]
        # Independent samples of quotients, not copies of source labels.
        profile = [cell[1][1] for cell in source[0]]
        cells = []
        for b in profile:
            pool = [g for g in G.elements if g != G.e and g[1] == b]
            cells.append([G.e, rng.choice(pool)])
        return [cells]

    # One degree-three column and two degree-two columns: total seven.
    # Vary the trinomial's quotient-coordinate pattern deliberately.
    patterns = ((0, 1), (0, 2), (1, 1), (1, 2),
                (1, 3), (2, 2), (2, 3), (2, 4))
    b0, b1 = patterns[(attempt // 4) % len(patterns)]
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    g = rng.choice(pools[b0])
    h = rng.choice(pools[b1])
    while h == g:
        h = rng.choice(pools[b1])
    cells = [[G.e, g, h]]
    for j in range(2):
        b = (attempt // (8 + 3 * j) + j) % 5
        cells.append([G.e, rng.choice(pools[b])])
    return [cells]


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


from functools import lru_cache


@lru_cache(maxsize=1)
def _annihilator_catalogue(G, count):
    """Deterministic computation returning only immutable selected cells.

    Singular-cell rank is a proposal feature, not a distance certificate.
    Full ranks, stabilizer membership and all syndromes use verified lifts.
    """
    from itertools import combinations
    from hashlib import sha256

    q, e, width = G.q, G.e, 3 * G.q
    assert q == 55
    cap = min(9, W_MAX)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            mark = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= mark
                row ^= bit
        return out

    def basis(rows):
        out = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                if p not in out:
                    out[p] = word
                    break
                word ^= out[p]
        return out

    def residue(word, own):
        for p in sorted(own, reverse=True):
            if (word >> p) & 1:
                word ^= own[p]
        return word

    def logical_four(checks, own, n):
        """Exact logical exclusion through four; short stabilizers are legal.

        Equal syndromes of words of size <=2 must have equal residues
        modulo the complete stabilizer space. One representative per
        syndrome suffices by linearity.
        """
        cols = columns(checks, n)
        pivots = sorted(own, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in pivots:
                if (word >> p) & 1:
                    word ^= own[p]
            residues.append(word)

        seen = {0: (0, 0)}
        for j, syndrome in enumerate(cols):
            old = seen.get(syndrome)
            r = residues[j]
            if old is None:
                seen[syndrome] = (r, 1 << j)
            elif old[0] != r:
                return None
        for j, first in enumerate(cols):
            bit, rj = 1 << j, residues[j]
            for i in range(j):
                syndrome = first ^ cols[i]
                r = rj ^ residues[i]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = (r, bit | (1 << i))
                elif old[0] != r:
                    return None
        return cols, residues, seen

    def has_logical_five(screen):
        """Exhaustive triple/pair test after logical_four passed.

        No symmetry of the complete nonabelian LP graph is assumed.
        A residue mismatch is a nontrivial logical; a match is harmless.
        """
        cols, residues, seen = screen
        for j, first in enumerate(cols):
            rj = residues[j]
            for i in range(j):
                syndrome = first ^ cols[i]
                r = rj ^ residues[i]
                for h in range(i):
                    old = seen.get(syndrome ^ cols[h])
                    if old is not None and old[0] != r ^ residues[h]:
                        return True
        return False

    def explicit_five(A, B, hx, hz, bx, bz):
        """Find a nontrivial embedded order-five binomial cycle, if present.

        These are actual supports in the verified column order. Neither
        commutation nor nontriviality is assumed from the element order.
        """
        for side, grid, checks, own in (
                ("A", A, hx, bz), ("B", B, hz, bx)):
            for block, cell in enumerate(grid[0]):
                if len(cell) != 2 or cell[0] != e:
                    continue
                power, orbit = e, []
                for _ in range(5):
                    orbit.append(G.idx(power))
                    power = G.mul(power, cell[1])
                if power != e or len(set(orbit)) != 5:
                    continue
                for other in range(3):
                    base = ((3 * block + other) if side == "A"
                            else (3 * other + block)) * q
                    word = sum(1 << (base + t) for t in orbit)
                    if any((row & word).bit_count() & 1 for row in checks):
                        continue
                    if residue(word, own):
                        return word
        return 0

    def basic_key(hx, hz):
        return sha256(repr(_tanner_signature(hx, hz)).encode("ascii")).digest()

    def full_key(hx, hz, n, basic):
        """Additional intrinsic invariants of the uncoloured Tanner graph."""
        rows = hx + hz
        cols = columns(rows, n)
        adjacency = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                if row & rows[j]:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        walks = tuple(sorted(
            (rows[i].bit_count(), adjacency[i].bit_count(),
             sum((adjacency[i] & other).bit_count() ** 2
                 for other in adjacency))
            for i in range(len(rows))))
        value = (
            basic, tuple(sorted(row.bit_count() for row in rows)),
            tuple(sorted(col.bit_count() for col in cols)), walks)
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key):
        return [[[e, G.elements[t]] for t in key]]

    # Include short-cycle and disconnected components deliberately.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != e
    })
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    actions = []
    for u in range(1, 11):
        for v in range(11):
            image = {}
            for t in reps:
                a, b = G.elements[t]
                g = ((u * a + v * sums[b]) % 11, b)
                image[t] = min(G.idx(g), G.idx(G.inv(g)))
            actions.append(image)
    keys = {
        min(tuple(sorted(image[t] for t in triple)) for image in actions)
        for triple in combinations(reps, 3)
    }
    bank = []
    for key in sorted(keys):
        A = binomials(key)
        rows = _component_rows(G, "A", A)
        reading, collisions = _kernel_profile(rows, width)
        if reading < 3:
            continue
        bank.append((gf2_rank(rows), reading, collisions, key, A))
    bank.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    clean = [r for r in bank if r[1] == 5]
    if not clean:
        raise ValueError("no conventional clean binomial fallback")

    # Exact ranks of ALL identity-anchored trinomials. A zero dummy
    # exposes the actual one-cell left action and is never published.
    pools = [[g for g in G.elements if g != e and g[1] == b]
             for b in range(5)]
    triangles = {}
    for g, h in combinations([g for g in G.elements if g != e], 2):
        cell = [e, g, h]
        rows, _, _ = lifted_matrices(G, [[cell]], [[[]]])
        rank = gf2_rank(rows)
        pattern = tuple(sorted((g[1], h[1])))
        triangles.setdefault(pattern, []).append(
            (rank, tuple(sorted((G.idx(g), G.idx(h)))), cell))
    for records in triangles.values():
        records.sort(key=lambda r: (r[0], r[1]))

    # Known current-parent classes, reconstructed from the shown table.
    # Equal basic invariants are rejected conservatively. This is NOT
    # access to the evaluator's entire, partly unshown archive.
    known_A = (
        ((1, 0), (0, 1), (5, 2)),
        ((1, 0), (0, 2), (5, 2)),
        ((0, 1), (1, 2), (10, 2)),
    )
    known_B = (
        (0, (10,0), (7,1), (3,0), (1,1)),
        (1, (5,0), (2,1), (6,0), (10,2)),
        (2, (4,0), (8,1), (4,0), (9,3)),
        (0, (4,0), (5,1), (7,0), (1,4)),
        (1, (10,0), (4,1), (1,1), (9,1)),
        (2, (2,0), (5,1), (2,1), (2,2)),
        (0, (6,0), (1,1), (10,1), (0,3)),
        (2, (5,0), (2,1), (8,2), (9,2)),
        (0, (1,0), (6,1), (6,2), (8,3)),
        (1, (6,0), (0,1), (9,2), (1,4)),
        (2, (5,0), (3,1), (0,3), (5,4)),
        (1, (10,0), (4,2), (4,0), (7,1)),
        (0, (8,0), (1,2), (2,0), (3,3)),
        (1, (5,0), (2,2), (0,3), (6,2)),
        (2, (8,0), (10,2), (3,1), (0,3)),
        (0, (3,0), (7,2), (6,2), (4,2)),
        (1, (5,0), (5,2), (10,2), (6,3)),
        (0, (5,0), (2,2), (0,3), (3,3)),
        (0, (4,0), (3,3), (6,0), (10,1)),
        (1, (5,0), (2,3), (8,0), (5,2)),
        (1, (9,0), (3,3), (5,1), (4,1)),
        (0, (8,0), (1,3), (7,1), (0,3)),
        (0, (2,0), (5,3), (4,3), (10,3)),
        (2, (10,0), (4,3), (2,4), (8,4)),
        (0, (2,0), (9,4), (4,0), (3,2)),
        (1, (3,0), (5,4), (6,0), (0,3)),
        (0, (1,0), (6,4), (5,1), (10,1)),
        (0, (1,0), (1,4), (7,1), (0,3)),
        (2, (4,0), (1,4), (10,2), (1,2)),
        (2, (10,0), (9,4), (8,3), (6,3)),
        (2, (6,1), (9,1), (9,0), (5,1)),
        (2, (10,0), (9,4), (8,1), (9,1)),
    )
    archived = set()
    for ai, g, h, s, t in known_B:
        A = [[[e, x] for x in known_A[ai]]]
        B = [[[e, g, h], [e, s], [e, t]]]
        hx, hz, _ = lifted_matrices(G, A, B)
        archived.add(basic_key(hx, hz))

    tested_fillings, tested_graphs = set(), set()
    entries = []

    def consider(A, B, card, minimum_k=222):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in tested_fillings:
            return None
        tested_fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > cap
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        bx, bz = basis(hx), basis(hz)
        k = n - len(bx) - len(bz)
        if n != 550 or k < minimum_k:
            return None
        if not (375 * n <= 1000 * k < 425 * n):
            return None

        sx = logical_four(hz, bx, n)
        if sx is None:
            return None
        sz = logical_four(hx, bz, n)
        if sz is None:
            return None

        basic = basic_key(hx, hz)
        if basic in archived:
            return None
        sig = full_key(hx, hz, n, basic)
        if sig in tested_graphs:
            return None
        tested_graphs.add(sig)

        witness = explicit_five(A, B, hx, hz, bx, bz)
        if witness:
            floor = 5
        elif has_logical_five(sx) or has_logical_five(sz):
            floor = 5
        else:
            # Both tests were exhaustive: this is a genuine lower bound,
            # not a large sentinel assigned after an unsuccessful probe.
            floor = 6
        del sx, sz

        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        quality = (k * floor * floor / float(n), floor, k)
        return (quality, freeze(A), freeze(B), k, floor, card, sig.hex())

    patterns = (
        (0,0), (0,1), (0,2), (0,3), (1,1),
        (1,2), (1,3), (1,4), (2,2), (2,3))
    local = random.Random(BASE_SEED ^ 0x55A661)

    # Twenty deliberate quotient/alignment prescriptions. First test
    # cyclic binomial annihilators against low-rank trinomial blocks.
    # Complete quantum screening, not constituent girth, decides.
    for card in range(20):
        records = triangles[patterns[card % 10]]
        singular = [r for r in records if r[0] < q]
        source = singular if singular else records
        for trial in range(48):
            tri_rank, _, cell = source[trial % len(source)]
            cyclic = local.choice(pools[0 if card < 10 else 1])
            exponent = 2 + (card + trial // 11) % (
                4 if card < 10 else 2)
            second = e
            for _ in range(exponent):
                second = G.mul(second, cyclic)
            if second in (e, cyclic, G.inv(cyclic)):
                continue
            A = bank[(trial // 3 + card) % len(bank)][4]
            B = [[cell, [e, cyclic], [e, second]]]
            answer = consider(
                A, B, ("annihilator", card, trial, tri_rank), 226)
            if answer is not None:
                entries.append(answer)
                if len(entries) >= count:
                    break
        if len(entries) >= count:
            break

    # Fresh, rank-legal fallback cards. Singular blocks remain preferred,
    # but unit blocks are not prohibited. Each card optimizes several
    # complete, independently screened candidates rather than one draw.
    reinforcement = ((0,1), (0,2), (1,1), (1,2), (2,2),
                     (1,3), (1,4), (2,3), (2,4), (3,4))
    while len(entries) < count:
        position = len(entries)
        card, epoch = position % 20, position // 20
        records = triangles[patterns[card % 10]]
        c0, c1 = reinforcement[(card + epoch) % len(reinforcement)]
        survivors = []
        for trial in range(1024):
            # Alternate low-rank priority and coverage of the whole
            # verified cell-rank catalogue.
            index = ((trial // 2 + 7 * epoch) % len(records)
                     if trial % 2 == 0 else local.randrange(len(records)))
            tri_rank, _, cell = records[index]
            s, t = local.choice(pools[c0]), local.choice(pools[c1])
            if s == t or s == G.inv(t):
                continue
            if trial % 5:
                if G.mul(s, t) == G.mul(t, s):
                    continue
                component = clean[(position + trial // 7) % len(clean)]
            else:
                component = bank[(position + trial // 3) % len(bank)]
            A = component[4]
            B = [[cell, [e, s], [e, t]]]

            # A cheap verified proposal screen, not the acceptance gate.
            br = _component_rows(G, "B", B)
            if _kernel_profile(br, width)[0] < 4:
                continue
            answer = consider(
                A, B, ("reinforcement", card, epoch, trial, tri_rank))
            if answer is None:
                continue
            survivors.append(answer)
            if answer[4] >= 6 or len(survivors) >= 3:
                break
        if not survivors:
            # Cache the valid prefix rather than throwing away earlier
            # successful cards or silently publishing an unscreened code.
            break
        entries.append(max(survivors, key=lambda r: (r[0], r[6])))

    entries.sort(key=lambda r: (r[0], r[6]), reverse=True)
    return tuple(entries)


@lru_cache(maxsize=1)
def _hubroot_catalogue(G, count):
    """Immutable portfolio; every accepted code has exact distance five.

    Stronger invariant matches cause conservative rejection. Unequal
    invariants certify inequivalence, but this is not a complete iso test.
    The earlier annihilator catalogue is inactive.
    """
    from itertools import combinations, combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    q, e = G.q, G.e
    assert q == 55
    width, cap = 3 * q, min(9, W_MAX)

    def columns(rows, n):
        out = [0] * n
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def basis(rows):
        out = {}
        for original in rows:
            word = original
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def reduce(word, own):
        for p in sorted(own, reverse=True):
            if (word >> p) & 1:
                word ^= own[p]
        return word

    def logical_four(checks, own, n):
        """Exact logical exclusion through four, permitting stabilizers."""
        cols = columns(checks, n)
        pivots = sorted(own, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in pivots:
                if (word >> p) & 1:
                    word ^= own[p]
            residues.append(word)

        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            residue = residues[j]
            if syndrome in seen:
                if seen[syndrome] != residue:
                    return False
            else:
                seen[syndrome] = residue

        for j, first in enumerate(cols):
            rj = residues[j]
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = rj ^ residues[i]
                if syndrome in seen:
                    if seen[syndrome] != residue:
                        return False
                else:
                    seen[syndrome] = residue
        # Every support of size <=4 splits into two supports of size <=2.
        return True

    def five_witness(A, B, hx, hz, bx, bz):
        """Actual commuting, nontrivial supports in verified coordinates."""
        for side, grid, checks, own in (
                ("A", A, hx, bz), ("B", B, hz, bx)):
            for block, cell in enumerate(grid[0]):
                if len(cell) != 2 or cell[0] != e:
                    continue
                power, orbit = e, []
                for _ in range(5):
                    orbit.append(G.idx(power))
                    power = G.mul(power, cell[1])
                if power != e or len(set(orbit)) != 5:
                    continue
                for other in range(3):
                    base = ((3 * block + other) if side == "A"
                            else (3 * other + block)) * q
                    word = sum(1 << (base + t) for t in orbit)
                    if any((row & word).bit_count() & 1 for row in checks):
                        continue
                    if reduce(word, own):
                        return ("Z" if side == "A" else "X"), word
        return None

    def basic_key(hx, hz, n):
        rows = hx + hz
        value = (
            tuple(sorted(row.bit_count() for row in rows)),
            tuple(sorted(c.bit_count() for c in columns(rows, n))),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def rooted_key(hx, hz, n):
        """Full bipartite Tanner refinement rooted at EVERY maximum hub.

        X and Z checks are not coloured differently. Check/qubit roles,
        degrees, and the set of maximum-degree qubits are intrinsic.
        The refinement trace records vocabularies as well as counts:
        locally assigned colour IDs alone would lose information.
        """
        rows = hx + hz
        m = len(rows)
        cols = columns(rows, n)
        neighbours = [[] for _ in range(m + n)]
        for i, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                j = bit.bit_length() - 1
                neighbours[i].append(m + j)
                neighbours[m + j].append(i)
                row ^= bit

        degrees = [len(adj) for adj in neighbours]
        maximum = max(c.bit_count() for c in cols)
        roots = [m + j for j, col in enumerate(cols)
                 if col.bit_count() == maximum]
        root_keys = []
        for root in roots:
            labels = [(int(v >= m), degrees[v], int(v == root))
                      for v in range(m + n)]
            vocabulary = sorted(set(labels))
            ids = {value: i for i, value in enumerate(vocabulary)}
            colours = [ids[value] for value in labels]
            initial = tuple(sorted(Counter(labels).items()))
            trace = [sha256(repr(initial).encode("ascii")).digest()]

            for _ in range(6):
                labels = [
                    (colours[v],
                     tuple(sorted(colours[u] for u in neighbours[v])))
                    for v in range(m + n)
                ]
                histogram = tuple(sorted(Counter(labels).items()))
                trace.append(
                    sha256(repr(histogram).encode("ascii")).digest())
                vocabulary = sorted(set(labels))
                ids = {value: i for i, value in enumerate(vocabulary)}
                colours = [ids[value] for value in labels]
            root_keys.append(sha256(b"".join(trace)).digest())
        return sha256(b"".join(sorted(root_keys))).digest()

    # Buckets hold explicit archived references and accepted matrices.
    # Expensive rooted invariants are evaluated lazily on collisions.
    buckets = {}

    def add_reference(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        key = basic_key(hx, hz, n)
        buckets.setdefault(key, []).append([hx, hz, n, None])

    aa = (
        ((1, 0), (0, 1), (5, 2)),
        ((1, 0), (0, 2), (5, 2)),
        ((0, 1), (1, 2), (10, 2)),
    )
    known_even = (
        (aa[0], ((1, 0), (0, 2), (5, 2))),
        (aa[0], aa[2]),
        (aa[1], aa[2]),
        (aa[0], ((2, 0), (0, 1), (10, 2))),
        (((9, 0), (2, 2), (3, 2)),
         ((10, 0), (8, 2), (3, 2))),
        (((10, 0), (1, 1), (9, 2)),
         ((10, 0), (10, 1), (1, 2))),
    )
    for a, b in known_even:
        add_reference([[[e, g] for g in a]],
                      [[[e, g] for g in b]])

    # Current measured examples are references, never proposal seeds.
    known_odd = (
        (0, (4, 0), (4, 2), (7, 1), (3, 1)),
        (0, (0, 1), (6, 1), (6, 1), (6, 3)),
        (0, (5, 0), (7, 0), (6, 0), (4, 2)),
        (1, (0, 1), (2, 4), (3, 2), (4, 3)),
        (2, (3, 0), (4, 0), (3, 1), (9, 1)),
    )
    for ai, g, h, s, t in known_odd:
        add_reference([[[e, x] for x in aa[ai]]],
                      [[[e, g, h], [e, s], [e, t]]])

    def novelty_token(hx, hz, n):
        key = basic_key(hx, hz, n)
        bucket = buckets.get(key)
        if not bucket:
            return key, None
        rooted = rooted_key(hx, hz, n)
        for record in bucket:
            if record[3] is None:
                record[3] = rooted_key(record[0], record[1], record[2])
            if record[3] == rooted:
                return None
        return key, rooted

    def reserve(token, hx, hz, n):
        key, rooted = token
        buckets.setdefault(key, []).append([hx, hz, n, rooted])

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key, scale=1):
        return [[[e, ((scale * G.elements[t][0]) % 11,
                      G.elements[t][1])] for t in key]]

    # Exhaustive orbit coverage: normalize a rotation to x, or an
    # order-five generator (up to inversion) to y or y^2.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != e
    })
    orbit_keys = set()
    for anchor in (G.idx((1, 0)), G.idx((0, 1)), G.idx((0, 2))):
        others = [t for t in reps if t != anchor]
        for pair in combinations(others, 2):
            orbit_keys.add(_generator_key(
                G, [G.elements[t] for t in (anchor,) + pair]))

    bank = []
    for key in sorted(orbit_keys):
        A = binomials(key)
        rows = _component_rows(G, "A", A)
        if gf2_rank(rows) != q - 1:
            continue
        if _kernel_profile(rows, width)[0] != 5:
            continue
        bank.append((key, A))
    if not bank:
        raise ValueError("no connected kernel-four-clean binomial component")

    tested = set()
    entries = []

    def inspect(A, B, card, required_k):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in tested:
            return None
        tested.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > cap
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        bx, bz = basis(hx), basis(hz)
        k = n - len(bx) - len(bz)
        if n != 550 or k != required_k:
            return None
        if not (375 * n <= 1000 * k < 425 * n):
            return None
        if not logical_four(hz, bx, n):
            return None
        if not logical_four(hx, bz, n):
            return None
        witness = five_witness(A, B, hx, hz, bx, bz)
        if witness is None:
            return None

        token = novelty_token(hx, hz, n)
        if token is None:
            return None
        assert css_ok(hx, hz)
        assert witness[1].bit_count() == 5
        assert max(row.bit_count() for row in hx + hz) <= cap
        entry = (
            freeze(A), freeze(B), k, card, witness,
            token[0].hex(), (k * 25 / float(n), 5, k),
        )
        return entry, token, hx, hz, n

    def publish(result):
        entry, token, hx, hz, n = result
        reserve(token, hx, hz, n)
        entries.append(entry)

    # All relative multipliers are trials. Inner/gauge copies and
    # archived examples are rejected, never counted as separate slots.
    plans = [(a, b, scale)
             for a in range(len(bank))
             for b in range(a, len(bank))
             for scale in range(1, 11)]
    plans.sort(key=lambda p: (p[0] == p[1], p[2], p[0], p[1]))
    for a, b, scale in plans:
        result = inspect(
            bank[a][1], binomials(bank[b][0], scale),
            ("even-recovery", a, b, scale), 226)
        if result is not None:
            publish(result)
            if len(entries) == count:
                return tuple(entries)

    # Twenty cards optimize different quotient/reinforcement pairings.
    # Additional epochs extend the trinomial quotient patterns.
    patterns = tuple(combinations_with_replacement(range(5), 2))
    reinforcements = (
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    pools = [[g for g in G.elements if g != e and g[1] == b]
             for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55B026)

    while len(entries) < count:
        position = len(entries)
        card, epoch = position % 20, position // 20
        b0, b1 = patterns[(card % 10 + 5 * epoch) % len(patterns)]
        c0, c1 = reinforcements[
            (card % 10 + 3 * (card // 10) + epoch) % 10]
        winner = None

        for batch in range(12):
            proposals = []
            for trial in range(64):
                g, h = local.choice(pools[b0]), local.choice(pools[b1])
                if g == h:
                    continue
                s, t = local.choice(pools[c0]), local.choice(pools[c1])
                if s == t or s == G.inv(t):
                    continue
                B = [[[e, g, h], [e, s], [e, t]]]
                br = _component_rows(G, "B", B)
                if gf2_rank(br) != q:
                    continue
                if _kernel_profile(br, width)[0] != 5:
                    continue

                # Verified constituent Tanner 4-cycle pressure selects
                # proposals, not a claimed quantum-distance estimate.
                pressure = 0
                for i, row in enumerate(br):
                    for other in br[:i]:
                        overlap = (row & other).bit_count()
                        pressure += overlap * (overlap - 1) // 2
                ai = (position + trial // 5 + batch) % len(bank)
                filling = tuple(
                    tuple(sorted(G.idx(x) for x in cell))
                    for cell in B[0])
                proposals.append(
                    ((pressure, filling, ai), bank[ai][1], B, trial))
            proposals.sort(key=lambda item: item[0])

            for quality, A, B, trial in proposals[:16]:
                result = inspect(
                    A, B, ("odd-card", card, epoch, batch, trial), 222)
                if result is not None:
                    winner = result
                    break
            if winner is not None:
                break

        if winner is None:
            # Preserve valid earlier slots; never weaken the quantum gate.
            break
        publish(winner)

    return tuple(entries)


@lru_cache(maxsize=1)
def _cubic_spectrum_portfolio(G, count):
    """Immutable structural portfolio; historical order-55 banks are inactive.

    All syndromes, ranks and witnesses use verified construction helpers.
    Exact short-word gates and heuristic larger-word probes are separate.
    """
    from itertools import combinations, combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    q, e, width = G.q, G.e, 2 * G.q
    cap = min(9, W_MAX)
    assert q == 110 and A_SHAPE == B_SHAPE == (1, 2)

    def columns(rows, n):
        out = [0] * n
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def basis(rows):
        out = {}
        for original in rows:
            word = original
            while word:
                p = word.bit_length() - 1
                if p not in out:
                    out[p] = word
                    break
                word ^= out[p]
        return out

    def nontrivial(word, own):
        while word:
            p = word.bit_length() - 1
            if p not in own:
                return True
            word ^= own[p]
        return False

    def clean_four(rows, n):
        """Exactly exclude every nonzero kernel support of weight <=4."""
        cols = columns(rows, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return False
        pairs = set()
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    def constituent_clean_six(rows):
        """Exact constituent exclusion through six, after clean_four.

        Every column has odd degree three, excluding odd kernel weights.
        The one-row regular lift acts transitively inside each column
        block. A six-word can therefore be translated to contain one of
        the two identity-coordinate anchors. Its anchored triple must
        collide with a different triple. This symmetry is NOT assumed
        for the complete LP code.
        """
        cols = columns(rows, width)
        if any(col.bit_count() != 3 for col in cols):
            return False
        anchored = {}
        for anchor in (G.idx(e), q + G.idx(e)):
            others = [j for j in range(width) if j != anchor]
            for a, b in combinations(others, 2):
                syndrome = cols[anchor] ^ cols[a] ^ cols[b]
                word = (1 << anchor) | (1 << a) | (1 << b)
                old = anchored.get(syndrome)
                if old is not None and old != word:
                    return False
                anchored[syndrome] = word
        for a in range(width):
            for b in range(a):
                pair = cols[a] ^ cols[b]
                for c in range(b):
                    old = anchored.get(pair ^ cols[c])
                    if old is not None:
                        word = (1 << a) | (1 << b) | (1 << c)
                        if old != word:
                            return False
        return True

    def cycle_penalty(rows, n):
        """Tanner four-cycle count plus a projection-triangle penalty."""
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i, neighbours in enumerate(adjacency):
            for j in range(i):
                if (neighbours >> j) & 1:
                    triangles += (
                        neighbours & adjacency[j] & ((1 << j) - 1)
                    ).bit_count()
        internal = sum(
            c.bit_count() * (c.bit_count() - 1)
            * (c.bit_count() - 2) // 6
            for c in columns(rows, n))
        # With four-cycles this is only a structural penalty.
        return four, max(0, triangles - internal)

    def constituent_key(rows):
        """Intrinsic check-projection walk invariant for a one-row lift.

        Every check root is equivalent under its regular group action.
        Root zero therefore represents the all-root histogram, unlike
        the complete nonabelian LP graph, where roots need not agree.
        """
        adjacency = [[] for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adjacency[i].append((j, overlap))
                    adjacency[j].append((i, overlap))
        vector = [0] * len(rows)
        vector[0] = 1
        features = [[] for _ in rows]
        diagonal = []
        for _ in range(10):
            following = [0] * len(rows)
            for i, value in enumerate(vector):
                if value:
                    for j, multiplicity in adjacency[i]:
                        following[j] += value * multiplicity
            vector = following
            diagonal.append(vector[0])
            for j, value in enumerate(vector):
                features[j].append(value)
        value = (
            tuple(diagonal),
            tuple(sorted(tuple(feature) for feature in features)),
            _tanner_signature(rows, []),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def probe(rows, n, local, tries, stabilizers=None):
        """Return an actual kernel/logical witness weight, not a floor.

        Random information sets and kernel-basis exchanges are heuristic.
        n+1 means unresolved if no nontrivial logical was recorded.
        """
        cols = columns(rows, n)
        bits = [1 << j for j in range(n)]
        own = basis(stabilizers) if stabilizers is not None else {}
        best, best_word = n + 1, 0

        def record(word):
            nonlocal best, best_word
            weight = word.bit_count()
            if not word or weight >= best:
                return
            if stabilizers is not None and not nontrivial(word, own):
                return
            best, best_word = weight, word

        for _ in range(tries):
            order = list(range(n))
            local.shuffle(order)
            pivots, kernel = {}, []
            for j in order:
                syndrome, word = cols[j], bits[j]
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    kernel.append(word)
                    record(word)

            for _ in range(2):
                local.shuffle(kernel)
                for i, word in enumerate(kernel):
                    weight = word.bit_count()
                    for j, other in enumerate(kernel):
                        if i == j:
                            continue
                        candidate = word ^ other
                        new_weight = candidate.bit_count()
                        if 0 < new_weight < weight:
                            word, weight = candidate, new_weight
                    kernel[i] = word
                    record(word)

            short = sorted(kernel, key=int.bit_count)[:18]
            for i, word in enumerate(short):
                for j in range(i):
                    pair = word ^ short[j]
                    record(pair)
                    for h in range(j):
                        record(pair ^ short[h])
            if best_word:
                starting = best_word
                for i, word in enumerate(short):
                    record(starting ^ word)
                    for other in short[:i]:
                        record(starting ^ word ^ other)
            if best <= 6:
                break
        return best

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured Tanner invariant; equal keys are merged."""
        rows = hx + hz
        value = (
            n,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(
                c.bit_count() for c in columns(rows, n)).items())),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def element_order(g):
        power = e
        for t in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return t
        raise ValueError("invalid metacyclic element order")

    orders = {g: element_order(g) for g in G.elements}
    spectra = ((10, 10, 11), (5, 5, 11),
               (5, 10, 10), (5, 5, 5))
    triangles = [[] for _ in spectra]
    for g, h in combinations([g for g in G.elements if g != e], 2):
        spectrum = tuple(sorted((
            orders[g], orders[h], orders[G.mul(G.inv(g), h)])))
        if spectrum in spectra:
            triangles[spectra.index(spectrum)].append((e, g, h))
    if any(not pool for pool in triangles):
        raise ValueError("required cubic spectrum absent")

    policies = tuple(combinations_with_replacement(range(4), 2))
    local = random.Random(BASE_SEED ^ 0x110C020)
    pools, component_keys = {}, set()
    serial = 0

    for policy in policies:
        proposals, fillings = [], set()
        # These are optimization trials for a structural spectrum card,
        # not separately published random slots.
        for trial in range(192):
            C = [[list(local.choice(triangles[kind])) for kind in policy]]
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for cell in C[0])
            if filling in fillings:
                continue
            fillings.add(filling)
            left = _component_rows(G, "A", C)
            if gf2_rank(left) != q or not clean_four(left, width):
                continue
            four, six_pressure = cycle_penalty(left, width)
            proposals.append((
                (24 * four + six_pressure, four, filling),
                C, left))
        proposals.sort(key=lambda item: item[0])

        survivors = []
        for cheap, C, left in proposals[:40]:
            if not constituent_clean_six(left):
                continue
            signature = constituent_key(left)
            if signature in component_keys:
                continue
            component_keys.add(signature)

            # Independently expose the evaluator's right action as well.
            right = _component_rows(G, "B", C)
            if gf2_rank(right) != q or not clean_four(right, width):
                continue
            da = probe(left, width, local, 28)
            db = probe(right, width, local, 12)
            reading = min(da, db)
            if reading < 8:
                continue
            survivors.append((
                reading, cheap[0], signature, serial, freeze(C),
                policy, (da, db)))
            serial += 1
            if len(survivors) >= 10:
                break
        survivors.sort(
            key=lambda item: (item[0], -item[1], item[2]), reverse=True)
        if survivors:
            pools[policy] = tuple(survivors[:8])

    if not pools:
        raise ValueError("no cubic constituent survived verified screens")

    available = sorted(pools)
    preferred = (
        ((0, 2), (0, 2)),
        ((0, 0), (0, 2)),
        ((0, 2), (2, 2)),
        ((0, 0), (0, 0)),
        ((2, 2), (2, 2)),
    )
    cards = [tuple(sorted(card)) for card in preferred
             if card[0] in pools and card[1] in pools]
    for card in combinations_with_replacement(available, 2):
        if card not in cards:
            cards.append(card)

    # Exclusion reference only; never used as a proposal or bank seed.
    Aref = [[
        [e, (5, 0), (9, 3)],
        [e, (6, 2), (1, 3)],
    ]]
    Bref = [[
        [e, (10, 0), (9, 1)],
        [e, (6, 6), (7, 7)],
    ]]
    rx, rz, rn = lifted_matrices(G, Aref, Bref)
    excluded = {graph_key(rx, rz, rn)}
    entries = []

    for position in range(count):
        winner = None
        # Alternate structural cards only if a card has exhausted its
        # invariant-distinct pairings; never weaken the validity gates.
        for retry in range(len(cards)):
            card_index = (position + retry) % len(cards)
            pa, pb = cards[card_index]
            plans = []
            for a in pools[pa]:
                for b in pools[pb]:
                    if pa == pb and a[3] > b[3]:
                        continue
                    pressure = 1 + position % 5
                    quality = (
                        min(a[0], b[0]),
                        -(pressure * a[1] + (6 - pressure) * b[1]),
                        -(a[1] + b[1]), a[2], b[2])
                    plans.append((quality, a, b))
            plans.sort(key=lambda item: item[0], reverse=True)

            tested = 0
            for _, a, b in plans:
                A, B = a[4], b[4]
                sa = [[len(cell) for cell in row] for row in A]
                sb = [[len(cell) for cell in row] for row in B]
                if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                    continue

                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 550 or not (175 * n <= 1000 * k < 225 * n):
                    continue
                if max(row.bit_count() for row in hx + hz) > cap:
                    continue
                if not clean_four(hx, n) or not clean_four(hz, n):
                    continue
                signature = graph_key(hx, hz, n)
                if signature in excluded:
                    continue

                dx = probe(hz, n, local, 8, hx)
                dz = probe(hx, n, local, 8, hz)
                # This capped reading is an optimization feature only.
                # Exact acceptance certifies absence through four.
                reading = min(a[0], b[0], dx, dz, 24)
                quality = (
                    k * reading * reading / float(n),
                    reading, -(a[1] + b[1]), signature)
                result = (
                    quality, A, B, k,
                    (card_index, pa, pb, a[3], b[3]),
                    (a[6], b[6], dx, dz), signature)
                if winner is None or quality > winner[0]:
                    winner = result
                tested += 1
                if tested >= 6:
                    break
            if winner is not None:
                break

        if winner is None:
            break
        quality, A, B, k, card, readings, signature = winner
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz), "verified cubic lift failed CSS"
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) <= cap
        excluded.add(signature)
        entries.append((
            A, B, k, card, readings, signature.hex(), quality[:3]))

    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish one immutable, structurally optimized target-rate design."""
    catalogue = _cubic_spectrum_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("novel cubic spectrum portfolio exhausted")
    A, B, k, card, readings, signature, quality = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "verified_complete_kernel_floor": 5,
            "design_card": card,
            "kernel_witness_probe_readings": readings,
            "selection_quality": quality,
            "tanner_signature": signature}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent screens and the two-second proxy suffice to protect the Frobenius-bed finalists from weight-four logicals; slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: Cross-pairing inequivalent Cayley profiles supplies five evaluator-distinct k=226 finalists: the current table supplied only three, slot 3 collapsed to slot 0, and the other two measured finalists had k=222.
# REFUTED: Singular trinomials and shared cyclic reinforcement raise this programme's measured finalists above k=222 or distance five: all five current measured finalists instead had k=222 and d30s=5.
# REFUTED: Maximum-degree-qubit-rooted refinement recovers a fresh k=226 finalist on the order-55 (1,3)x(1,3) bed: all 39 distinct current candidates had k=222, and all five measured finalists retained d30s=5.
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
