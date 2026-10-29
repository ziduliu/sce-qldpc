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
CONCEPT = ("On C11 semidirect_3 C5, order 55, A=(1,3), B=(1,3), rank-deficient constituents whose short dependencies become complete-code stabilizers may supply fresh k>=226, distance-at-least-five designs and improve cell 0.400; stabilizer shielding is a falsifiable hypothesis.")
GROUP = ("Frozen L1 group implementation and order, kept byte-for-byte. This bed leads cell 0.400 near fitness 10.2; the current program scored 10.1274. Orders five and eleven explain the observed binomial cycle scales, not a universal distance ceiling. Retain the measured bed and search a different rank/homology mechanism.")
SHAPES = ("Frozen equal (1,3) sides, kept byte-for-byte: constituent variables/checks 165/55, n=550, kb=220 and guaranteed rate 0.400; the size window permits q<=59. Current minimum logicals occupy sector-1 Acol/Bcol blocks; slots 3..6 concentrate the X family at Bcol2 across all A columns. Earlier weight-four failures also touched Arow0/Brow0. Verified complete ranks retain [0.375,0.425).")
FILLING = ("Enumerate binomial triples with replacement, including disconnected and short-kernel constituents previously rejected. Also enumerate identity-anchored trinomials reinforced by commuting binomials and retain verified rank-deficient proposals. Explicit embedded short words must be stabilizers; both complete matrices then pass exact nontrivial-logical exclusion through four. Rank-first exploration is bounded, with twenty fresh trinomial/binomial fallback cards optimized under the same quantum gate. Current-table Tanner classes are reconstructed for conservative exclusion; distinct intrinsic signatures screen within-program repeats. Rank-surplus improvement and unseen archive novelty remain hypotheses. Immutable memoization affects build cost only.")
FAMILY_NAME = "r040_frobenius11by5_13x13_rankshield_quantum4_v1"

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


def _quantum4_parent_reference(rng, G, idx):
    """Uncalled historical catalogue; never publishes this program's cells."""
    from itertools import combinations

    local = random.Random(BASE_SEED ^ 0x5504E1)
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            mark = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= mark
                row ^= bit
        return out

    def quantum_clean_four(checks, stabilizers, width):
        """Exact exclusion of nontrivial logical supports of weight <=4.

        Every such support splits into two words of weight <=2.
        For each syndrome, all enumerated words must have the same
        residue modulo the stabilizer space. Short stabilizers are legal.
        """
        basis = {}
        for original in stabilizers:
            word = original
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        pivots = sorted(basis, reverse=True)

        def reduce(word):
            for p in pivots:
                if (word >> p) & 1:
                    word ^= basis[p]
            return word

        syndromes = columns(checks, width)
        residues = [reduce(1 << j) for j in range(width)]
        seen = {0: 0}
        for j, syndrome in enumerate(syndromes):
            residue = residues[j]
            if syndrome in seen:
                if residue != seen[syndrome]:
                    return False
            else:
                seen[syndrome] = residue
        for j, syndrome in enumerate(syndromes):
            residue = residues[j]
            for i in range(j):
                s = syndrome ^ syndromes[i]
                r = residue ^ residues[i]
                if s in seen:
                    if r != seen[s]:
                        return False
                else:
                    seen[s] = r
        return True

    def binomials(key):
        return [[[G.e, G.elements[t]] for t in key]]

    def image(key, u, v=0):
        result = []
        for t in key:
            a, b = G.elements[t]
            g = ((u * a + v * sums[b]) % 11, b)
            result.append(min(G.idx(g), G.idx(G.inv(g))))
        return tuple(sorted(result))

    # Every triple can be mapped to one containing one of these anchors:
    # a rotation maps to x; otherwise a generator, inverted if necessary,
    # has quotient coordinate 1 or 2 and its translation can be removed.
    undirected = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != G.e
    })
    anchors = [G.idx(g) for g in ((1, 0), (0, 1), (0, 2))]
    orbit_keys = set()
    for anchor in anchors:
        others = [t for t in undirected if t != anchor]
        for pair in combinations(others, 2):
            generators = [G.elements[t] for t in (anchor,) + pair]
            orbit_keys.add(_generator_key(G, generators))

    legacy = {entry[1] for entry in _component_bank(G)}
    bank = []
    for key in sorted(orbit_keys):
        A = binomials(key)
        ar = _component_rows(G, "A", A)
        if _kernel_profile(ar, 3 * G.q)[0] != 5:
            continue
        # Connected component incidence has rank q-1; disconnected
        # graphs are not allowed to win merely through rank deficits.
        if gf2_rank(ar) != G.q - 1:
            continue
        br = _component_rows(G, "B", A)
        if _kernel_profile(br, 3 * G.q)[0] != 5:
            continue
        bank.append(key)
    bank.sort(key=lambda key: (key in legacy, key))
    if not bank:
        raise ValueError("exhaustive binomial bank has no clean component")

    def signature(hx, hz, n):
        cols = columns(hx + hz, n)
        return (
            tuple(sorted(row.bit_count() for row in hx + hz)),
            tuple(sorted(c.bit_count() for c in cols)),
            _tanner_signature(hx, hz),
        )

    graph_keys = set()
    # Explicit archived pure-binomial examples from the current table
    # and the live frontier. This is conservative invariant rejection,
    # not a claim to have access to the entire evaluator archive.
    archived = (
        (((9, 0), (2, 2), (3, 2)),
         ((10, 0), (8, 2), (3, 2))),
        (((10, 0), (1, 1), (9, 2)),
         ((10, 0), (10, 1), (1, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((1, 0), (0, 2), (5, 2))),
    )
    for aa, bb in archived:
        A = [[[G.e, g] for g in aa]]
        B = [[[G.e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(signature(hx, hz, n))

    filling_keys = set()
    accepted = 0

    def inspect(A, B):
        nonlocal accepted
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (375 * n <= 1000 * k < 425 * n):
            return None
        sig = signature(hx, hz, n)
        if sig in graph_keys:
            return None
        # Reject this entire invariant class after its first inspection.
        # Different accepted signatures certify non-isomorphic graphs.
        graph_keys.add(sig)
        if not quantum_clean_four(hz, hx, n):
            return None
        if not quantum_clean_four(hx, hz, n):
            return None
        if accepted == idx:
            assert n == 550 and k > 0
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "quantum_logical_floor": 5,
                    "support": (tuple(sa[0]), tuple(sb[0]))}
        accepted += 1
        return None

    # Cover new component-family pairings first. Relative automorphism
    # images are only trials: Tanner invariants discard orbit copies.
    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    plans.sort(key=lambda ab: (
        int(bank[ab[0]] in legacy) + int(bank[ab[1]] in legacy),
        ab[0] == ab[1], ab))
    for a, b in plans:
        for u in range(1, 11):
            result = inspect(binomials(bank[a]),
                             binomials(image(bank[b], u)))
            if result is not None:
                return result

    # Asymmetric supports remain a fallback, not a competitor against
    # available k=226, weight-eight candidates. Their component and
    # full quantum screens are equally strict.
    descriptors = [((5, 0), key, binomials(key)) for key in bank]
    for attempt in range(4096):
        if attempt % 4 == 0:
            continue
        A = binomials(bank[attempt % len(bank)])
        B = _draw_B(local, G, attempt, descriptors)
        if _kernel_profile(
                _component_rows(G, "B", B), 3 * G.q)[0] != 5:
            continue
        result = inspect(A, B)
        if result is not None:
            return result
    raise ValueError("novel quantum-four-clean catalogue exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _rankshield_portfolio(G, count):
    """Immutable rank-first portfolio using only verified matrix lifts."""
    from itertools import combinations, combinations_with_replacement
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

    def basis(rows):
        result = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = result.get(p)
                if old is None:
                    result[p] = word
                    break
                word ^= old
        return result

    def outside(word, own):
        while word:
            p = word.bit_length() - 1
            old = own.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def short_word(rows, n):
        """Return an actual kernel support through four, or zero."""
        cols = columns(rows, n)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            word = 1 << j
            old = seen.get(syndrome)
            if old is not None:
                return old ^ word
            seen[syndrome] = word
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                word = (1 << j) | (1 << i)
                old = seen.get(syndrome)
                if old is not None:
                    difference = old ^ word
                    if difference:
                        return difference
                else:
                    seen[syndrome] = word
        return 0

    def logical_four(checks, own, n):
        """Exactly exclude logicals <=4, permitting short stabilizers."""
        pivots = sorted(own, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in pivots:
                if (word >> p) & 1:
                    word ^= own[p]
            residues.append(word)
        cols = columns(checks, n)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            r = residues[j]
            old = seen.get(syndrome)
            if old is not None:
                if old != r:
                    return False
            else:
                seen[syndrome] = r
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                r = residues[j] ^ residues[i]
                old = seen.get(syndrome)
                if old is not None:
                    if old != r:
                        return False
                else:
                    seen[syndrome] = r
        return True

    def embedded_bad(word, side, hx, hz, bx, bz):
        """Reject only verified, nontrivial embedded short witnesses."""
        if not word:
            return False
        checks, own = (hx, bz) if side == "A" else (hz, bx)
        for block in range(3):
            if side == "B":
                embedded = word << (block * width)
            else:
                embedded, bits = 0, word
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    embedded |= 1 << ((3 * j + block) * q + h)
                    bits ^= bit
            # Embedding conventions are checked, never presumed.
            if any((row & embedded).bit_count() & 1 for row in checks):
                continue
            if outside(embedded, own):
                return True
        return False

    def graph_key(hx, hz, n):
        rows = hx + hz
        value = (
            n, tuple(sorted(row.bit_count() for row in rows)),
            tuple(sorted(c.bit_count() for c in columns(rows, n))),
            _tanner_signature(hx, hz))
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key, scale=1):
        return [[[e, (scale * G.elements[t][0] % 11,
                       G.elements[t][1])] for t in key]]

    # Repeated directions are legal across different cells. They are
    # proposals, not automatically accepted as useful degeneracy.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != e
    })
    orbit_keys = {
        _generator_key(G, [G.elements[t] for t in triple])
        for triple in combinations_with_replacement(reps, 3)
    }
    bank = []
    for key in sorted(orbit_keys):
        A = binomials(key)
        ar = _component_rows(G, "A", A)
        br = _component_rows(G, "B", A)
        rank = gf2_rank(ar)
        aw, bw = short_word(ar, width), short_word(br, width)
        bank.append((key, A, rank, aw, bw))
    bank.sort(key=lambda r: (
        bool(r[3] or r[4]), r[2] != q - 1, -r[2], r[0]))
    clean = [r for r in bank if not r[3] and not r[4]]
    if not clean:
        raise ValueError("no clean binomial fallback constituent")

    # Exclusion references from the CURRENT table, never candidate seeds.
    known_A = (
        ((1,0), (0,1), (5,2)),
        ((1,0), (0,2), (5,2)),
        ((0,1), (1,2), (10,2)),
    )
    known_B = (
        (1, (4,0), (6,1), (10,0), (10,1)),
        (2, (4,0), (6,1), (6,0), (2,1)),
        (0, (6,0), (1,1), (7,0), (0,1)),
        (2, (7,0), (2,2), (1,0), (0,1)),
        (0, (4,1), (7,1), (7,1), (3,1)),
        (2, (4,1), (8,1), (0,1), (2,2)),
        (1, (7,1), (8,2), (7,1), (4,2)),
        (0, (3,1), (6,2), (4,1), (6,2)),
        (2, (4,1), (7,3), (8,2), (5,2)),
        (0, (7,1), (7,3), (0,2), (1,2)),
        (1, (0,1), (0,3), (7,2), (2,2)),
        (0, (8,2), (9,2), (2,2), (10,2)),
        (1, (3,2), (10,2), (9,2), (3,3)),
        (2, (4,2), (7,2), (7,2), (10,3)),
        (2, (4,2), (8,3), (10,3), (9,3)),
        (0, (10,2), (7,3), (0,3), (5,3)),
        (0, (10,2), (10,4), (10,3), (5,3)),
        (1, (7,2), (2,4), (1,3), (5,3)),
        (2, (6,0), (0,1), (2,4), (1,4)),
        (1, (1,0), (1,2), (6,4), (5,4)),
        (2, (5,0), (7,2), (0,4), (8,4)),
        (0, (2,0), (2,2), (1,4), (0,4)),
        (2, (1,1), (6,1), (6,0), (3,4)),
        (1, (10,1), (5,3), (2,1), (10,0)),
        (2, (0,1), (1,3), (0,1), (10,0)),
        (0, (2,2), (9,2), (9,1), (7,0)),
        (1, (4,2), (7,3), (5,2), (0,1)),
        (2, (4,2), (7,3), (10,2), (3,1)),
        (2, (10,2), (1,4), (0,2), (7,1)),
        (0, (7,2), (6,4), (7,2), (2,1)),
        (0, (5,0), (5,1), (5,3), (5,2)),
        (1, (2,0), (8,1), (4,3), (2,2)),
        (1, (3,1), (10,1), (3,4), (3,2)),
        (2, (7,1), (6,2), (6,4), (4,3)),
        (0, (7,1), (1,2), (2,4), (2,3)),
        (0, (6,1), (7,3), (1,0), (8,3)),
        (1, (9,1), (4,3), (7,0), (5,3)),
        (2, (7,1), (4,3), (5,0), (8,3)),
        (2, (3,2), (7,3), (2,1), (4,4)),
        (0, (9,2), (6,3), (2,1), (5,4)),
        (1, (6,2), (9,3), (10,1), (8,4)),
    )
    graph_keys = set()
    for ai, g, h, s, t in known_B:
        A = [[[e, x] for x in known_A[ai]]]
        B = [[[e, g, h], [e, s], [e, t]]]
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(graph_key(hx, hz, n))

    known_pairs = (
        (known_A[0], known_A[1]),
        (known_A[0], known_A[2]),
        (known_A[1], known_A[2]),
        (known_A[0], ((2,0), (0,1), (10,2))),
        (((9,0), (2,2), (3,2)),
         ((10,0), (8,2), (3,2))),
        (((10,0), (1,1), (9,2)),
         ((10,0), (10,1), (1,2))),
    )
    for aa, bb in known_pairs:
        hx, hz, n = lifted_matrices(
            G, [[[e, g] for g in aa]], [[[e, g] for g in bb]])
        graph_keys.add(graph_key(hx, hz, n))

    fillings, entries = set(), []
    full_trials = 0

    def consider(A, B, card, minimum_k, aw=0, bw=0):
        nonlocal full_trials
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            return None
        fillings.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return None
        full_trials += 1
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or k < minimum_k:
            return None
        if not (375 * n <= 1000 * k < 425 * n):
            return None
        bx, bz = basis(hx), basis(hz)
        if embedded_bad(aw, "A", hx, hz, bx, bz):
            return None
        if embedded_bad(bw, "B", hx, hz, bx, bz):
            return None
        signature = graph_key(hx, hz, n)
        if signature in graph_keys:
            return None
        if not logical_four(hz, bx, n):
            return None
        if not logical_four(hx, bz, n):
            return None
        graph_keys.add(signature)
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) <= cap
        return ((k * 25 / float(n), k), freeze(A), freeze(B),
                k, card, signature.hex())

    # Test conventional and degenerate pairings, not only girth-five
    # constituent pairs. Relative outer orientations are actual trials.
    plans = [(a, b, scale)
             for a in range(len(bank))
             for b in range(a, len(bank))
             for scale in (1, 2)]
    plans.sort(key=lambda p: (
        bool(bank[p[0]][3] or bank[p[1]][4]),
        bank[p[0]][2] + bank[p[1]][2], p))
    for a, b, scale in plans:
        if full_trials >= 512 or len(entries) >= count:
            break
        A = bank[a][1]
        B = binomials(bank[b][0], scale)
        bw = short_word(_component_rows(G, "B", B), width)
        answer = consider(
            A, B, ("degenerate-binomial", a, b, scale),
            226, bank[a][3], bw)
        if answer is not None:
            entries.append(answer)

    # Canonical commuting-pair prescriptions: normal-C11 pairs and
    # complement-C5 pairs, including repeated directions.
    commuting = (
        ((1,0), (1,0)), ((1,0), (2,0)),
        ((1,0), (3,0)), ((1,0), (4,0)),
        ((1,0), (5,0)), ((0,1), (0,1)),
        ((0,1), (0,2)), ((0,2), (0,2)),
    )
    deficient = []
    nonzero = [g for g in G.elements if g != e]
    for card, (s, t) in enumerate(commuting):
        for g, h in combinations(nonzero, 2):
            B = [[[e, g, h], [e, s], [e, t]]]
            br = _component_rows(G, "B", B)
            rank = gf2_rank(br)
            if 50 <= rank < q:
                bw = short_word(br, width)
                deficient.append((rank, card, G.idx(g), G.idx(h), B, bw))
    deficient.sort(key=lambda r: r[:4])

    for position, record in enumerate(deficient):
        if full_trials >= 2048 or len(entries) >= count:
            break
        rank, card, _, _, B, bw = record
        # Cover clean and low-rank A prescriptions with equal trial caps.
        ids = list(range(min(4, len(clean))))
        ids += [(position + 7 * card + j) % len(bank) for j in range(4)]
        for ai in dict.fromkeys(ids):
            if full_trials >= 2048:
                break
            answer = consider(
                bank[ai][1], B,
                ("rank-shield", card, position, ai, rank),
                226, bank[ai][3], bw)
            if answer is not None:
                entries.append(answer)
                if len(entries) >= count:
                    break

    # Twenty deliberate fallback cards: triangle quotients, binomial
    # reinforcement quotients, constituent class and outer orientation.
    patterns = (
        (0,0), (0,1), (0,2), (0,3), (1,1),
        (1,2), (1,3), (1,4), (2,2), (2,3))
    reinforcement = (
        (0,1), (0,2), (1,1), (1,2), (2,2),
        (1,3), (1,4), (2,3), (2,4), (3,4))
    pools = [[g for g in nonzero if g[1] == b] for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55A9B41)
    component_cache = {}
    supplementary = 0
    while len(entries) < count:
        card, epoch = supplementary % 20, supplementary // 20
        b0, b1 = patterns[card % 10]
        c0, c1 = reinforcement[(card // 2 + 3 * epoch) % 10]
        survivors = []
        for trial in range(2048):
            g, h = local.choice(pools[b0]), local.choice(pools[b1])
            s, t = local.choice(pools[c0]), local.choice(pools[c1])
            if g == h:
                continue
            B = [[[e, g, h], [e, s], [e, t]]]
            fingerprint = tuple(
                tuple(sorted(G.idx(x) for x in cell)) for cell in B[0])
            profile = component_cache.get(fingerprint)
            if profile is None:
                br = _component_rows(G, "B", B)
                profile = (gf2_rank(br), short_word(br, width))
                component_cache[fingerprint] = profile
            if profile[1]:
                continue
            ai = (card + epoch + trial // 11) % len(clean)
            scale = 1 if card < 10 else 2
            A = binomials(clean[ai][0], scale)
            answer = consider(
                A, B, ("fresh-fallback", card, epoch, ai, scale, trial),
                220)
            if answer is not None:
                survivors.append(answer)
                if len(survivors) == 4:
                    break
        if not survivors:
            break
        entries.append(max(survivors, key=lambda r: (r[0], r[5])))
        supplementary += 1

    entries.sort(key=lambda r: (r[0], r[5]), reverse=True)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish one immutable, complete-homology-screened design."""
    portfolio = _rankshield_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("fresh rank-shield portfolio exhausted")
    quality, A, B, k, card, signature = portfolio[idx]
    return {"A": A, "B": B, "screen_k": k,
            "quantum_logical_floor": 5,
            "design_card": card, "selection_quality": quality,
            "tanner_signature": signature}


def CELL(rng, G, side, i, j, ctx):
    """Publish the selected full cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: The Frobenius component-screened catalogue consistently gives d30s>4: parent slot 30 fell from d2s=5 to d30s=4, with both X and Z logicals touching both LP sectors.
# REFUTED: Exhaustive clean-binomial enumeration supplies five fresh k=226 finalists: the current table measured only one fresh k=226 finalist, archived slots 0 and 1, and filled the other four measured positions with k=222,d30s=5 codes.
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
