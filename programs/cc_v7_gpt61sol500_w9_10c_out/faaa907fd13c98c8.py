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
CONCEPT = ("On C11 semidirect_3 C5, order 55, A=B=(1,3), exact-weighted check-walk invariants through length four should recover previously merged inequivalent kernel-four-clean binomial pairings and supply fresh k=226 finalists in cell 0.400; this is a falsifiable novelty hypothesis.")
GROUP = ("Frozen L1 group and order, kept byte-for-byte. This bed leads cell 0.400 at fitness 10.2002; the current singular-reinforcement filling scored 10.0910 with five measured k=222,d30s=5 finalists. Orders five and eleven motivate binomial cycle scales, not quantum-distance guarantees. Retain the successful bed and test a stronger novelty selector.")
SHAPES = ("Frozen equal (1,3) sides, kept byte-for-byte: each constituent has 165 variables and 55 checks, n=550, kb=220 and guaranteed rate 0.400; the size window permits q<=59. Parent minima occupied sector-1 Acol/Bcol blocks broadly, including all A columns at Bcol1/2. Earlier weight-four failures also touched Arow0/Brow0, so both complete matrices are screened. Verified ranks retain [0.375,0.425).")
FILLING = ("Enumerate every canonical connected binomial constituent and test every relative scaling. Joint weighted two-, three- and four-walk profiles distinguish complete Tanner graphs without treating an old coarse-signature collision as equivalence. Exclude six explicitly shown archived binomial pairings and selected shown trinomial examples. All outputs exclude complete kernel words through four and pass verified rank, CSS and weight gates. Remaining slots optimize twenty fresh trinomial/binomial quotient cards with different constituent and cycle-multiplicity pressures. Additional fresh k=226 classes and unseen archive novelty remain hypotheses; GPU distance readings remain upper bounds.")
FAMILY_NAME = "r040_frobenius11by5_13x13_weighted_jointwalk4_fresh"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    # x^11=y^5=1, y*x*y^-1=x^3; 3 has order five modulo 11.
    if not valid_q(q):
        raise ValueError("this affine group has order 55")
    powers = [pow(3, b, 11) for b in range(5)]
    els = [(a, b) for b in range(5) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 5)

    def mul(g, h):
        return ((g[0] + powers[g[1] % 5] * h[0]) % 11,
                (g[1] + h[1]) % 5)

    def inv(g):
        bi = (-g[1]) % 5
        return ((-powers[bi] * g[0]) % 11, bi)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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
    """Extract the component from the verified construction.

    A one-cell dummy side reduces screening cost without changing the
    left/right action used for the component being extracted.
    """
    dummy = [[[G.e]]]
    mask = (1 << (3 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx[:G.q]]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz[:G.q]]


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
def _weighted_walk_catalogue(G, count):
    """Immutable portfolio; all matrix screens use the verified lift.

    Equal invariant keys are conservatively merged. Unlike the previous
    selector, an equal coarse check-intersection key is not itself a veto.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, e, width = G.q, G.e, 3 * G.q
    cap = min(9, W_MAX)
    assert q == 55

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def clean_four(rows, n):
        """Exact exclusion of all nonzero kernel supports through four."""
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

    def graph_key(hx, hz, n):
        """Joint weighted check-walk invariant of the uncoloured Tanner graph.

        The projection edge weight is the EXACT number of shared qubits,
        not the older one-versus-several approximation. All check roots
        are included; no full-LP transitivity is assumed.
        """
        rows = hx + hz
        m = len(rows)
        degrees = [row.bit_count() for row in rows]
        cols = columns(rows, n)
        qdegrees = [col.bit_count() for col in cols]
        incidence = [[] for _ in rows]
        for j, col in enumerate(cols):
            bits = col
            while bits:
                bit = bits & -bits
                incidence[bit.bit_length() - 1].append(qdegrees[j])
                bits ^= bit
        labels = [
            (degrees[i], tuple(sorted(Counter(incidence[i]).items())))
            for i in range(m)
        ]

        adjacency = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                shared = (row & rows[j]).bit_count()
                if shared:
                    adjacency[i][j] = shared
                    adjacency[j][i] = shared
        neighbours = [tuple(a.items()) for a in adjacency]

        square = [[0] * m for _ in rows]
        for i in range(m):
            target = square[i]
            for h, first in neighbours[i]:
                for j, second in neighbours[h]:
                    target[j] += first * second

        def multiply(matrix):
            result = [[0] * m for _ in rows]
            for i in range(m):
                target = result[i]
                for h, multiplier in neighbours[i]:
                    for j, value in enumerate(matrix[h]):
                        if value:
                            target[j] += multiplier * value
            return result

        cube = multiply(square)
        fourth = multiply(cube)
        profiles = Counter()
        for i in range(m):
            histogram = Counter(
                (labels[j], adjacency[i].get(j, 0),
                 square[i][j], cube[i][j], fourth[i][j])
                for j in range(m))
            profile = (
                labels[i], square[i][i], cube[i][i], fourth[i][i],
                tuple(sorted(histogram.items())))
            profiles[profile] += 1
        value = (
            n, m,
            tuple(sorted(Counter(degrees).items())),
            tuple(sorted(Counter(qdegrees).items())),
            tuple(sorted(profiles.items())))
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key, scale=1):
        return [[[e, (scale * G.elements[t][0] % 11,
                       G.elements[t][1])] for t in key]]

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

    orbit_keys = {
        min(tuple(sorted(image[t] for t in triple))
            for image in actions)
        for triple in combinations(reps, 3)
    }
    bank = []
    for key in sorted(orbit_keys):
        A = binomials(key)
        ar = _component_rows(G, "A", A)
        if gf2_rank(ar) != q - 1 or not clean_four(ar, width):
            continue
        br = _component_rows(G, "B", A)
        if gf2_rank(br) != q - 1 or not clean_four(br, width):
            continue
        bank.append((key, A))
    if not bank:
        raise ValueError("no connected kernel-four-clean binomial bank")

    # References are only exclusion examples, never candidate outputs.
    # This is not access to the evaluator's unshown archive.
    known_pairs = (
        (((9,0), (2,2), (3,2)),
         ((10,0), (8,2), (3,2))),
        (((10,0), (1,1), (9,2)),
         ((10,0), (10,1), (1,2))),
        (((1,0), (0,1), (5,2)),
         ((1,0), (0,2), (5,2))),
        (((1,0), (0,1), (5,2)),
         ((0,1), (1,2), (10,2))),
        (((1,0), (0,2), (5,2)),
         ((0,1), (1,2), (10,2))),
        (((1,0), (0,1), (5,2)),
         ((2,0), (0,1), (10,2))),
    )
    excluded = set()
    for aa, bb in known_pairs:
        A = [[[e, g] for g in aa]]
        B = [[[e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(graph_key(hx, hz, n))

    # Also exclude the five measured fillings of the current program.
    known_triangles = (
        (((1,0), (0,1), (5,2)),
         ((4,0), (4,2), (7,1), (3,1))),
        (((1,0), (0,1), (5,2)),
         ((0,1), (6,1), (6,1), (6,3))),
        (((1,0), (0,1), (5,2)),
         ((5,0), (7,0), (6,0), (4,2))),
        (((1,0), (0,2), (5,2)),
         ((0,1), (2,4), (3,2), (4,3))),
        (((0,1), (1,2), (10,2)),
         ((3,0), (4,0), (3,1), (9,1))),
    )
    for aa, (g, h, s, t) in known_triangles:
        A = [[[e, x] for x in aa]]
        B = [[[e, g, h], [e, s], [e, t]]]
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(graph_key(hx, hz, n))

    fillings, entries = set(), []

    def consider(A, B, card):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            return None
        fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return None
        if any(len({G._key(g) for g in cell}) != len(cell)
               for grid in (A, B) for row in grid for cell in row):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or not (375*n <= 1000*k < 425*n):
            return None
        if not clean_four(hx, n) or not clean_four(hz, n):
            return None
        signature = graph_key(hx, hz, n)
        if signature in excluded:
            return None
        excluded.add(signature)
        assert k > 0 and css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        assert lifted_k(G, A, B) == k
        quality = (k * 25 / float(n), k)
        return (quality, freeze(A), freeze(B), k, card, signature.hex())

    # Do not assume that an untested relative orientation is an orbit
    # copy. Its actual complete Tanner invariant and kernel gate decide.
    primary = []
    for ai, (_, A) in enumerate(bank):
        for bi in range(ai, len(bank)):
            for scale in range(1, 11):
                result = consider(
                    A, binomials(bank[bi][0], scale),
                    ("weighted-binomial", ai, bi, scale))
                if result is not None:
                    primary.append(result)
    primary.sort(key=lambda r: (r[0], r[5]), reverse=True)
    entries.extend(primary[:count])

    patterns = (
        (0,0), (0,1), (0,2), (0,3), (1,1),
        (1,2), (1,3), (1,4), (2,2), (2,3))
    reinforcement = (
        (0,1), (0,2), (1,1), (1,2), (2,2),
        (1,3), (1,4), (2,3), (2,4), (3,4))
    pools = [[g for g in G.elements if g != e and g[1] == b]
             for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55D4A7)
    component_cache = {}
    supplementary = 0

    while len(entries) < count:
        card, epoch = supplementary % 20, supplementary // 20
        b0, b1 = patterns[card % 10]
        c0, c1 = reinforcement[(card // 2 + 3*epoch) % 10]
        survivors = []
        for trial in range(2048):
            g, h = local.choice(pools[b0]), local.choice(pools[b1])
            s, t = local.choice(pools[c0]), local.choice(pools[c1])
            if g == h or s == t or s == G.inv(t):
                continue
            B = [[[e, g, h], [e, s], [e, t]]]
            key = tuple(tuple(sorted(G.idx(x) for x in cell))
                        for cell in B[0])
            profile = component_cache.get(key)
            if profile is None:
                rows = _component_rows(G, "B", B)
                profile = (gf2_rank(rows), clean_four(rows, width))
                component_cache[key] = profile
            if not profile[1]:
                continue

            # Cards vary the constituent class and the relative outer
            # orientation as well as the triangle and binomial quotients.
            ai = (card + epoch + trial // 13) % len(bank)
            scale = 1 if card < 10 else 2
            A = binomials(bank[ai][0], scale)
            result = consider(
                A, B, ("weighted-trinomial", card, epoch, ai,
                       scale, trial, profile[0]))
            if result is not None:
                survivors.append(result)
                if len(survivors) >= 3:
                    break
        if not survivors:
            break
        entries.append(max(survivors, key=lambda r: (r[0], r[5])))
        supplementary += 1

    entries.sort(key=lambda r: (r[0], r[5]), reverse=True)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish a full filling from the immutable weighted-walk portfolio."""
    catalogue = _weighted_walk_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("fresh weighted-walk catalogue exhausted")
    quality, A, B, k, card, signature = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "quantum_logical_floor": 5,
            "design_card": card, "selection_quality": quality,
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
# REFUTED: Singular trinomials and shared cyclic-binomial annihilators raise the emitted rank surplus or logical floor on this bed: all 48 distinct current candidates had k=222,d2s=5, and all five measured finalists had d30s=5.
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
