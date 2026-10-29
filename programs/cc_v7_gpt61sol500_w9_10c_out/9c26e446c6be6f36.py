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
CONCEPT = ("On split-metacyclic C11 semidirect_1 C1, order 11, A=B=(2,7), grounding BOTH incidence constituents should supply five novel exact-distance-four codes with k=275 in cell 0.450, improving that cell's current fitness rather than the parent's lower-rate fitness.")
GROUP = ("Keep the split-metacyclic construction rule, resizing only m,r,s to 11,1,1. Factor 53 permits q<=11, so order eleven is the largest admissible cyclic lift. This cyclic (2,7) bed measured distance four, but reported k=280 or 287 generally placed it in cell 0.500. Grounding both constituents tests a different full-rank mechanism; no larger-distance advantage is attributed to the group.")
SHAPES = ("Change both sides to (2,7): each constituent has 77 variables and 22 checks, n=583, kb=275 and guaranteed rate 25/53=0.471698, inside the requested cell 0.450. Verified complete ranks require k=275, preventing rank surplus from crossing 0.475. Parent minima were sector-1 Acol/Bcol witnesses broadly, especially around Acol2/3; this proposal trades its distance-six profile for the requested higher rate.")
FILLING = ("Exhaustively enumerate grounded six-matching constituents and five-matching constituents with an opposite-layer ring. Row totals are six/seven and column totals one/two, giving BOTH CSS maxima nine. Verified left/right constituents must have full rank and no kernel words through three. Final verified matrices exclude complete kernels through three and verify an actual non-stabilizer weight-four witness, certifying distance exactly four. Common affine orbit keys and intrinsic full-Tanner contraction invariants conservatively reject repeats. Five novel [583,275,4] measurements score 7.54717 versus the target cell's current 7.47826; novelty against the unseen archive remains evaluator-tested.")
FAMILY_NAME = "r045_m11r1s1_27x27_doubleground_fullrank_d4_catalog_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same split metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    # The new size factor 53 permits the maximal cyclic lift q=11.
    m, r, s = 11, 1, 1
    if not valid_q(q):
        raise ValueError("this semidirect product has order eleven")
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

A_SHAPE = (2, 7)               # (ra, ca)
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


def _component_rows(G, side, grid):
    """Extract every component row using the verified lift conventions."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz]


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
def _hyperedge_catalogue(G, count):
    """Optimize once; return immutable, invariant-distinct screened codes."""
    from itertools import permutations
    from collections import Counter
    from hashlib import sha256

    assert G.q == 8
    q, width, cap = G.q, 8 * G.q, min(9, W_MAX)
    local = random.Random(BASE_SEED ^ 0x38B605)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def pairs(cols):
        """Exact rejection of nonzero kernel words of weights 1..4."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        out = {}
        for j, first in enumerate(cols):
            for i in range(j):
                value = first ^ cols[i]
                if value in singles or value in out:
                    return None
                out[value] = (i, j)
        return out

    def six_count(rows):
        """Exclude weights <=5 and count all weight-six kernel supports."""
        cols = columns(rows, width)
        table = pairs(cols)
        if table is None:
            return None
        triples, collisions = {}, 0
        for j, first in enumerate(cols):
            for i in range(j):
                value = first ^ cols[i]
                for h in range(i):
                    syndrome = value ^ cols[h]
                    if syndrome in table:
                        return None
                    old = triples.get(syndrome, 0)
                    collisions += old
                    triples[syndrome] = old + 1
        # With weights <=5 excluded, equal-syndrome triples are disjoint.
        # Each six-support has ten unordered triple/triple partitions.
        assert collisions % 10 == 0
        return collisions // 10

    def profile_key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(8)))
            for order in permutations(range(3)))

    seed = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    profiles, profile_keys = [seed], {profile_key(seed)}
    doubles = sorted(set(permutations((1, 1, 0)))
                     | set(permutations((2, 0, 0))))
    triples = sorted(set(permutations((2, 1, 0)))
                     | {(1, 1, 1)}
                     | set(permutations((3, 0, 0))))
    for attempt in range(65536):
        cols = [(1, 0, 0)]
        cols += [local.choice(doubles) for _ in range(4)]
        cols += [local.choice(triples) for _ in range(3)]
        profile = tuple(tuple(c[i] for c in cols) for i in range(3))
        if any(sum(row) != 6 for row in profile):
            continue
        key = profile_key(profile)
        if key in profile_keys:
            continue
        profile_keys.add(key)
        profiles.append(profile)
        if len(profiles) == 20:
            break

    def draw(profile, measured_core):
        C = [[[] for _ in range(8)] for _ in range(3)]
        if measured_core:
            C[0][0] = [G.e]
            C[0][1] = C[1][1] = [G.e]
            for j, voltage in enumerate((0, 1, 3), 2):
                C[1][j] = [G.e]
                C[2][j] = [G.elements[voltage]]
            for j, row in ((5, 0), (6, 0), (7, 1)):
                C[row][j] = [G.e, G.elements[local.randrange(1, q)]]
                C[2][j] = [G.elements[local.randrange(q)]]
            return C
        for j in range(8):
            anchor = None
            for i in range(3):
                size = profile[i][j]
                if not size:
                    continue
                labels = local.sample(range(q), size)
                if anchor is None:
                    anchor = labels[0]
                C[i][j] = [
                    G.elements[t] for t in
                    sorted((label - anchor) % q for label in labels)]
        return C

    bank, fillings = [], set()
    quota = Counter()
    for attempt in range(16384):
        pi = 0 if attempt % 3 == 0 else (
            1 + (attempt // 3) % max(1, len(profiles) - 1))
        pi %= len(profiles)
        if quota[pi] >= (24 if pi == 0 else 4):
            continue
        C = draw(profiles[pi], pi == 0)
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in C for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        if not _no_dead_lines(counts, counts):
            continue
        assert all(sum(row) == 6 for row in counts)
        assert max(sum(row[j] for row in counts) for j in range(8)) <= 3

        left = _component_rows(G, "A", C)
        right = _component_rows(G, "B", C)
        if gf2_rank(left) != 24 or gf2_rank(right) != 24:
            continue
        nl = six_count(left)
        if nl is None:
            continue
        nr = six_count(right)
        if nr is None:
            continue
        quota[pi] += 1
        bank.append((max(nl, nr), nl + nr, filling, C, pi))
        if len(bank) >= 48:
            break
    if len(bank) < 8:
        raise ValueError("too few kernel-five-clean hyperedge constituents")
    bank.sort(key=lambda item: item[:3])

    def signature(hx, hz, n):
        rows = hx + hz
        cols = columns(rows, n)
        check_degrees = [row.bit_count() for row in rows]
        qubit_profiles = Counter()
        for col in cols:
            neighbours, bits = [], col
            while bits:
                bit = bits & -bits
                neighbours.append(check_degrees[bit.bit_length() - 1])
                bits ^= bit
            qubit_profiles[tuple(sorted(neighbours))] += 1
        value = (
            tuple(sorted(check_degrees)),
            tuple(sorted(qubit_profiles.items())),
            _tanner_signature(hx, hz))
        return sha256(repr(value).encode("ascii")).digest()

    def reference(gaps, tails):
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = [G.e]
        C[0][1] = C[1][1] = [G.e]
        for j, voltage in enumerate((0, 1, 3), 2):
            C[1][j] = [G.e]
            C[2][j] = [G.elements[voltage]]
        for j, row, gap, tail in zip(
                (5, 6, 7), (0, 0, 1), gaps, tails):
            C[row][j] = [G.e, G.elements[gap]]
            C[2][j] = [G.elements[tail]]
        return C

    # Exclusion references only; never emitted. The unseen archive remains
    # authoritative, and equal invariants are discarded conservatively.
    r0 = reference((6, 1, 1), (2, 7, 6))
    r1 = reference((1, 2, 3), (7, 4, 5))
    r2 = reference((7, 3, 5), (5, 7, 7))
    r3 = reference((5, 6, 5), (2, 4, 7))
    r4 = reference((6, 1, 5), (2, 7, 7))
    excluded = set()
    for A, B in ((r0, r0), (r0, r1), (r0, r2),
                 (r1, r1), (r3, r4)):
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(signature(hx, hz, n))

    def mixed_five(cols, table, n1):
        # Pure sector-1 words are constituent kernel words, already
        # excluded through five. Every remaining five-support has a
        # triple partition containing a sector-2 qubit.
        for j in range(n1, len(cols)):
            first = cols[j]
            for i in range(j):
                value = first ^ cols[i]
                for h in range(i):
                    if (value ^ cols[h]) in table:
                        return True
        return False

    plans = sorted(
        (int(bank[a][0] != 0 or bank[b][0] != 0),
         max(bank[a][0], bank[b][0]),
         bank[a][1] + bank[b][1],
         int(bank[a][4] == bank[b][4]), a, b)
        for a in range(len(bank)) for b in range(a, len(bank)))
    entries = []
    for _, _, _, _, a, b in plans:
        A, B = bank[a][3], bank[b][3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (325 * n <= 1000 * k < 375 * n):
            continue
        sig = signature(hx, hz, n)
        if sig in excluded:
            continue

        legal = True
        for checks in (hx, hz):
            cols = columns(checks, n)
            table = pairs(cols)
            if table is None or mixed_five(cols, table, 64 * q):
                legal = False
                break
        if not legal:
            continue
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) <= cap
        excluded.add(sig)

        freeze = lambda grid: tuple(
            tuple(tuple(cell) for cell in row) for row in grid)
        entries.append((
            freeze(A), freeze(B), k,
            (bank[a][0], bank[b][0]),
            (bank[a][4], bank[b][4]), sig.hex()))
        if len(entries) == count:
            break
    return tuple(entries)


@lru_cache(maxsize=1)
def _doubleground_catalogue(G, count):
    """Pure exhaustive structural catalogue with immutable publication.

    All algebraic screens use the scaffold's verified matrices.
    Distinct intrinsic signatures certify distinct Tanner graphs;
    equality is inconclusive and is discarded conservatively.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    if G.q != 11:
        raise ValueError("double-grounded incidence catalogue requires q=11")
    q, width, cap = G.q, 7 * G.q, min(9, W_MAX)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def short_info(rows, n):
        """Exclude weights 1..3; count fours and retain a four-witness."""
        cols = columns(rows, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return None
        representatives, multiplicities = {}, {}
        collisions, witness = 0, 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles:
                    return None
                word = (1 << i) | (1 << j)
                old = representatives.get(syndrome)
                if old is not None:
                    difference = word ^ old
                    assert difference.bit_count() == 4
                    if not witness:
                        witness = difference
                else:
                    representatives[syndrome] = word
                previous = multiplicities.get(syndrome, 0)
                collisions += previous
                multiplicities[syndrome] = previous + 1
        # Distinct nonzero columns make pair collisions disjoint.
        # Every four-support has three unordered pair partitions.
        assert collisions % 3 == 0
        return collisions // 3, witness

    def clean_three(rows, n):
        cols = columns(rows, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return False
        return all(first ^ cols[j] not in singles
                   for i, first in enumerate(cols) for j in range(i))

    def normalized(values):
        """Independent translations/reflections are conservative gauges."""
        return min(
            tuple(sorted((sign * (v - anchor)) % q for v in values))
            for anchor in values for sign in (1, -1))

    def image(card, scale):
        kind, step, values = card
        values = normalized(tuple(scale * v % q for v in values))
        step = min(scale * step % q, -scale * step % q)
        return kind, step, values

    def pair_orbit(a, b):
        # Remove common cyclic automorphisms and interchange of the sides.
        return min(tuple(sorted((image(a, u), image(b, u))))
                   for u in range(1, q))

    def build(card):
        kind, step, values = card
        if kind == 0:
            # Six cross-matchings plus a grounded upper layer.
            C = [
                [[G.e] for _ in values] + [[G.e]],
                [[G.elements[v]] for v in values] + [[]],
            ]
        else:
            # No upper-layer ring: such an edge plus two grounding
            # variables would immediately produce a weight-three word.
            C = [
                [[G.e] for _ in values] + [[G.e], []],
                [[G.elements[v]] for v in values]
                + [[], [G.e, G.elements[step]]],
            ]
        assert len(C) == 2 and all(len(row) == 7 for row in C)
        counts = [[len(cell) for cell in row] for row in C]
        assert sorted(sum(row) for row in counts) == [6, 7]
        assert max(sum(row[j] for row in counts) for j in range(7)) == 2
        return C

    cards = {
        (0, 0, normalized(values))
        for values in combinations(range(q), 6)
    }
    # A maximum independent set of an eleven-cycle excludes triangles
    # involving its lower-layer ring and two matching columns.
    for step in range(1, (q + 1) // 2):
        values = normalized(tuple(step * t % q for t in (0, 2, 4, 6, 8)))
        signed = {step, (-step) % q}
        if not any((a - b) % q in signed for a, b in combinations(values, 2)):
            cards.add((1, step, values))

    bank = []
    for card in sorted(cards):
        C = build(card)
        left = _component_rows(G, "A", C)
        right = _component_rows(G, "B", C)
        if gf2_rank(left) != 2 * q or gf2_rank(right) != 2 * q:
            continue
        li, ri = short_info(left, width), short_info(right, width)
        if li is None or ri is None or not ri[1]:
            continue
        bank.append((card, C, max(li[0], ri[0]), ri[1]))
    if len(bank) < 5:
        raise ValueError("too few full-rank grounded incidence cards")

    def graph_signature(hx, hz, n):
        """Intrinsic marked multigraph contraction of the full Tanner graph.

        Sector-2 qubits are identified ONLY by their actual degree >=12.
        Every check meets one or two such qubits. Contracting those
        checks gives edges or marked loops, labelled by check degree.
        Any full Tanner isomorphism preserves this construction.
        Rooted joint walk profiles use no protograph or group labels.
        """
        rows = hx + hz
        cols = columns(rows, n)
        high = [j for j, col in enumerate(cols) if col.bit_count() >= 12]
        if len(high) != 4 * q:
            raise ValueError("unexpected grounded contraction size")
        positions = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        size = len(high)
        adjacency = [Counter() for _ in high]
        relations = {}
        anchors = [Counter() for _ in high]
        incident = [Counter() for _ in high]

        for row in rows:
            bits, weight = row & high_mask, row.bit_count()
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("grounded check is not an edge or anchor")
            bit = bits & -bits
            u = positions[bit.bit_length() - 1]
            incident[u][weight] += 1
            if degree == 1:
                anchors[u][weight] += 1
                continue
            v = positions[(bits ^ bit).bit_length() - 1]
            incident[v][weight] += 1
            adjacency[u][v] += 1
            adjacency[v][u] += 1
            relation = relations.setdefault(weight, [Counter() for _ in high])
            relation[u][v] += 1
            relation[v][u] += 1

        labels = [
            (cols[j].bit_count(), tuple(sorted(anchors[v].items())),
             tuple(sorted(incident[v].items())))
            for v, j in enumerate(high)]
        graphs = [adjacency] + [relations[w] for w in sorted(relations)]
        neighbours = [
            [tuple(sorted(edges.items())) for edges in graph]
            for graph in graphs]
        root_profiles = []
        for root in range(size):
            endpoint_walks = [[] for _ in high]
            closed_walks = []
            for graph in neighbours:
                vector = [0] * size
                vector[root] = 1
                diagonal = []
                for length in range(1, 9):
                    following = [0] * size
                    for u, amount in enumerate(vector):
                        if amount:
                            for v, multiplicity in graph[u]:
                                following[v] += amount * multiplicity
                    vector = following
                    diagonal.append(vector[root])
                    for v, amount in enumerate(vector):
                        endpoint_walks[v].append(amount)
                closed_walks.append(tuple(diagonal))
            root_profiles.append((
                labels[root], tuple(closed_walks),
                tuple(sorted(
                    (labels[v], tuple(endpoint_walks[v]))
                    for v in range(size)))))

        qubit_profiles = Counter()
        weights = [row.bit_count() for row in rows]
        for col in cols:
            profile = []
            while col:
                bit = col & -col
                profile.append(weights[bit.bit_length() - 1])
                col ^= bit
            qubit_profiles[tuple(sorted(profile))] += 1
        signature = (
            n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(qubit_profiles.items())),
            tuple(sorted(relations)), tuple(sorted(root_profiles)))
        return sha256(repr(signature).encode("ascii")).digest()

    def nontrivial(word, stabilizers):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = row
                    break
                row ^= previous
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    plans = sorted(
        (max(bank[a][2], bank[b][2]), bank[a][2] + bank[b][2],
         int(bank[a][0][0] == bank[b][0][0]), a, b)
        for a in range(len(bank)) for b in range(a, len(bank)))
    entries, orbits, signatures = [], set(), set()
    for _, _, _, a, b in plans:
        orbit = pair_orbit(bank[a][0], bank[b][0])
        if orbit in orbits:
            continue
        orbits.add(orbit)
        A, B = bank[a][1], bank[b][1]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 583 or k != 275:
            continue
        if not (425 * n <= 1000 * k < 475 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not clean_three(hx, n) or not clean_three(hz, n):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified double-grounded lift failed CSS")

        # Embed an actual right-B four-word at A column zero.
        # Verify both commutation and stabilizer nonmembership explicitly.
        witness = bank[b][3]
        if any((row & witness).bit_count() & 1 for row in hz):
            raise ValueError("verified grounded four-witness embedding failed")
        if witness.bit_count() != 4 or not nontrivial(witness, hx):
            continue

        signature = graph_signature(hx, hz, n)
        if signature in signatures:
            continue
        signatures.add(signature)
        assert lifted_k(G, A, B) == k

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        entries.append((
            freeze(A), freeze(B), k, (bank[a][0], bank[b][0]),
            (bank[a][2], bank[b][2]), signature.hex()))
        if len(entries) >= count:
            break
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish one exact-distance-four grounded structural design."""
    catalogue = _doubleground_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("inequivalent double-grounded catalogue exhausted")
    A, B, k, cards, fours, signature = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "verified_distance": 4, "design_cards": cards,
            "constituent_four_counts": fours,
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
# REFUTED: Singular-annihilator reinforcement supplies measured rank-reserve or distance-six finalists: all five current finalists had k=222 and d30s=5, despite the rank-reserve-first proposal stage.
# REFUTED: Balanced hyperedge count cards and constituent six-pressure produce distance-seven finalists on the cyclic order-eight (3,8) bed: all five measured finalists instead had k=200,d30s=6.
# REFUTED: The balanced-hyperedge catalogue's reserved intrinsic signatures supply broad evaluator-distinct novelty: sixty slots retained only seven distinct codes, two additional archive duplicates, and many isomorphic repeats.
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
