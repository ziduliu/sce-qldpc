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
CONCEPT = ("UT3(F3) times C2, order 54, A=(1,3), B=(1,3): independently optimized order-six binomials and trinomial reinforcement should supply five fresh d30s=6 finalists in cell 0.400; mixed-sector distance survival is a falsifiable measurement hypothesis.")
GROUP = ("Jump from the order-55 Frobenius group to the order-54 direct product of the ternary Heisenberg group and C2. Noncentral elements with nontrivial C2 coordinate have order six. The parent measured only distance-five finalists, with abundant sector-1 witnesses. The new algebra removes its order-five binomial cycle scale, but does not guarantee quantum distance six. This exact group/shape bed is not identified in BEDS.")
SHAPES = ("Retain equal (1,3) constituents: 162 variables and 54 checks each, n=540, kb=216 and guaranteed rate 0.400. The shape permits q<=59. Parent weight-five minima occupied all nine sector-1 Acol/Bcol blocks, sometimes concentrating at Bcol2; earlier weight-four failures also touched Arow0/Brow0. Actual full ranks retain [0.375,0.425), and both complete logical spaces are screened.")
FILLING = ("Twenty primary cards combine five intrinsic trinomial prescriptions with four noncommuting quotient-direction pairs; additional epochs and bounded broader retries are optimized, not emitted as draws. A has three order-six binomials; B has one trinomial and two order-six binomials. HX<=8 and HZ<=9 satisfy both row budgets, the scaffold cap and the assignment cap twelve. Prefer constituents excluding kernels through five, enforce exact full-CSS logical exclusion through four, and select actual nontrivial witnesses after stabilizer reduction. Different intrinsic uncoloured Tanner invariants separate published slots. Immutable memoization changes build cost only. Constituent floors do not certify mixed-sector quantum distance, and unseen-archive novelty remains evaluator-tested.")
FAMILY_NAME = "r040_heisenberg3_times_c2_q54_13x13_order6_reinforcement"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("UT3(F3) times C2 has order 54")
    # (a,b,c) encodes [[1,a,c],[0,1,b],[0,0,1]] over F3.
    # The last coordinate is an independent central C2 factor.
    els = [(a, b, c, t)
           for t in range(2) for c in range(3)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3,
                int(g[2]) % 3, int(g[3]) % 2)

    def mul(g, h):
        a, b, c, t = red(g)
        d, e, f, u = red(h)
        return ((a + d) % 3, (b + e) % 3,
                (c + f + a * e) % 3, (t + u) % 2)

    def inv(g):
        a, b, c, t = red(g)
        return ((-a) % 3, (-b) % 3, (-c + a * b) % 3, t)

    def idx(g):
        a, b, c, t = red(g)
        return ((3 * t + c) * 3 + b) * 3 + a

    return dict(elements=els, ident=(0, 0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
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


def _unused_frobenius_catalogue(rng, G, idx):
    """Inactive historical reference; the order-54 publisher never calls it."""
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


def _h54_columns(rows, width):
    """Column syndromes, exclusively from scaffold-built matrices."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _h54_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _h54_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _h54_pairs(rows, width):
    """Exact kernel exclusion through four; return unique pair syndromes."""
    cols = _h54_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = {}
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = (1 << j) | (1 << i)
    return cols, pairs


def _h54_five(screen):
    """Actual weight-five kernel witness, or exhaustive exclusion.

    Called only after kernel weights one through four were excluded.
    Every five-support has a triple/pair partition.
    """
    cols, pairs = screen
    for j, first in enumerate(cols):
        bj = 1 << j
        for i in range(j):
            syndrome = first ^ cols[i]
            pair = bj | (1 << i)
            for h in range(i):
                old = pairs.get(syndrome ^ cols[h])
                if old is not None:
                    word = pair ^ (1 << h) ^ old
                    assert word.bit_count() == 5
                    return word
    return 0


def _h54_quantum4(checks, stabilizers, width):
    """Exact logical exclusion through four, permitting short stabilizers."""
    basis = _h54_basis(stabilizers)
    pivots = sorted(basis, reverse=True)
    residues = []
    for j in range(width):
        word = 1 << j
        for p in pivots:
            if (word >> p) & 1:
                word ^= basis[p]
        residues.append(word)

    cols = _h54_columns(checks, width)
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        residue = residues[j]
        if syndrome in seen:
            if seen[syndrome] != residue:
                return False
        else:
            seen[syndrome] = residue
    for j, first in enumerate(cols):
        residue = residues[j]
        for i in range(j):
            syndrome = first ^ cols[i]
            value = residue ^ residues[i]
            if syndrome in seen:
                if seen[syndrome] != value:
                    return False
            else:
                seen[syndrome] = value
    return True


def _h54_key(rows, width):
    """Intrinsic check/qubit Tanner invariant; equal keys merge conservatively.

    Actual overlap multiplicities and neighboring check degrees enrich
    the parent's intersection invariant. No group labels enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlap = {}
    for i, row in enumerate(rows):
        for j in range(i):
            c = (row & rows[j]).bit_count()
            if c:
                overlap[i, j] = c
                masks = one if c == 1 else many
                masks[i] |= 1 << j
                masks[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlap.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(weights[j],) + feature] += 1
            profiles[j][(weights[i],) + feature] += 1

    features = [
        (weights[i], tuple(sorted(profiles[i].items())))
        for i in range(m)]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    qubits = Counter()
    for col in _h54_columns(rows, width):
        neighborhood = []
        while col:
            bit = col & -col
            neighborhood.append(colors[bit.bit_length() - 1])
            col ^= bit
        qubits[tuple(sorted(neighborhood))] += 1
    value = (tuple(sorted(Counter(features).items())),
             tuple(sorted(qubits.items())))
    return sha256(repr(value).encode("ascii")).digest()


def _h54_binword(G, grid, rows):
    """Propose an order-six cycle, then verify against the actual component."""
    q = G.q
    for j, cell in enumerate(grid[0]):
        if len(cell) != 2 or cell[0] != G.e:
            continue
        g = cell[1]
        power, orbit = G.e, []
        for _ in range(6):
            orbit.append(G.idx(power))
            power = G.mul(power, g)
        if power != G.e or len(set(orbit)) != 6:
            continue
        word = sum(1 << (j * q + t) for t in orbit)
        if all((row & word).bit_count() % 2 == 0 for row in rows):
            return word
    return 0


def _h54_probe(rng, rows, width, passes, seed=0, own=None):
    """Actual kernel/logical witnesses, never unresolved distance sentinels."""
    cols = _h54_columns(rows, width)
    basis = _h54_basis(own) if own is not None else None
    best = 0

    def inspect(word):
        nonlocal best
        if not word or (best and word.bit_count() >= best.bit_count()):
            return
        if basis is not None and not _h54_outside(word, basis):
            return
        best = word

    if seed:
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)
        inspect(seed)

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        short = sorted(kernel, key=int.bit_count)[:32]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 16:
                    for h in range(j):
                        inspect(pair ^ short[h])

        if own is not None and best:
            for _ in range(2):
                starting = best
                moves = sorted(
                    own, key=lambda row: -(row & starting).bit_count())[:48]
                for i, first in enumerate(moves):
                    inspect(starting ^ first)
                    for second in moves[:i]:
                        inspect(starting ^ first ^ second)
                if best == starting:
                    break
        if best and best.bit_count() <= 5:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert basis is None or _h54_outside(best, basis)
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _h54_portfolio(G, count):
    """Immutable, deterministic portfolio of optimized structural cards."""
    from itertools import combinations

    if G.q != 54:
        raise ValueError("Heisenberg reinforcement requires order 54")
    q, width, e = G.q, 3 * G.q, G.e
    directions = ((1, 0), (0, 1), (1, 1), (1, 2))
    triples = tuple(combinations(range(4), 3))
    binpairs = ((0, 1), (0, 2), (1, 3), (2, 3))
    local = random.Random(BASE_SEED ^ 0x543C206)

    def draw_element(rng, direction, parity):
        a, b = directions[direction]
        scale = rng.choice((1, 2))
        return (scale * a % 3, scale * b % 3,
                rng.randrange(3), parity)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    bank, component_keys = [], set()
    for attempt in range(768):
        profile = triples[attempt % len(triples)]
        generators = [draw_element(local, d, 1) for d in profile]
        A = [[[e, g] for g in generators]]
        rows = _component_rows(G, "A", A)
        if gf2_rank(rows) != q - 1:
            continue
        screen = _h54_pairs(rows, width)
        if screen is None:
            continue
        # All incidence columns have degree two, and every edge changes
        # the C2 coordinate: odd cycles are excluded. The explicit
        # order-six word supplies the matching constituent upper bound.
        seed = _h54_binword(G, A, rows)
        if not seed:
            continue
        key = _h54_key(rows, width)
        if key in component_keys:
            continue
        component_keys.add(key)
        bank.append((freeze(A), seed, key))
        if len(bank) >= 16:
            break
    if not bank:
        raise ValueError("no connected order-six binomial constituent")

    def reinforced(rng, position, batch, trial):
        card, epoch = position % 20, position // 20
        primary_mode, policy = divmod(card, 4)
        # Preserve the primary motif for the first half of the budget;
        # broader retries avoid declaring an unsuccessful motif legal.
        mode = primary_mode if batch < 4 else (primary_mode + batch - 3) % 5
        d0 = (policy + epoch + trial // 8) % 4
        d1 = (d0 + 1 + policy % 3) % 4
        if mode == 0:
            g, h = draw_element(rng, d0, 1), draw_element(rng, d1, 1)
        elif mode == 1:
            g, h = draw_element(rng, d0, 0), draw_element(rng, d1, 1)
        elif mode == 2:
            g, h = draw_element(rng, d0, 0), draw_element(rng, d1, 0)
        else:
            g = draw_element(rng, d0, 1 if mode == 3 else 0)
            h = draw_element(rng, d0, 1)
            if h == g:
                h = (h[0], h[1], (h[2] + 1) % 3, h[3])
        if g == h:
            return None
        sdir, tdir = binpairs[(policy + epoch + batch // 2) % 4]
        s, t = draw_element(rng, sdir, 1), draw_element(rng, tdir, 1)
        assert G.mul(s, t) != G.mul(t, s)
        return [[[e, g, h], [e, s], [e, t]]], mode

    def embedded(source, kind, checks, stabilizers):
        own = _h54_basis(stabilizers)
        best = 0
        for block in range(3):
            if kind == "X":
                word = source << (block * width)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((3 * j + block) * q + t)
                    bits ^= bit
            # Embeddings are verified, not assumed from a hand-coded lift.
            if any((row & word).bit_count() % 2 for row in checks):
                continue
            if not _h54_outside(word, own):
                continue
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, published, fillings = [], set(), set()
    for position in range(count):
        rng = random.Random(BASE_SEED ^ (0x54C206 + 104729 * position))
        winner = None
        for batch in range(8):
            proposals = []
            for trial in range(96):
                draw = reinforced(rng, position, batch, trial)
                if draw is None:
                    continue
                B, mode = draw
                ai = (position + batch + trial // 5) % len(bank)
                A, aseed, _ = bank[ai]
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for grid in (A, B) for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                br = _component_rows(G, "B", B)
                screen = _h54_pairs(br, width)
                if screen is None:
                    continue
                rank = gf2_rank(br)
                bseed = _h54_binword(G, B, br)
                if not bseed:
                    continue
                proposals.append((rank, trial, A, B, aseed, bseed,
                                  br, screen, mode))
                if len(proposals) >= 8:
                    break

            refined = []
            for rank, trial, A, B, aseed, bseed, br, screen, mode in proposals:
                five = _h54_five(screen)
                floor = 5 if five else 6
                if five:
                    bseed = five
                refined.append((floor, rank, -trial, A, B,
                                aseed, bseed, mode))
            refined.sort(key=lambda item: item[:3], reverse=True)

            tested = 0
            for floor, rank, negtrial, A, B, aseed, bseed, mode in refined:
                sa = [[len(cell) for cell in row] for row in A]
                sb = [[len(cell) for cell in row] for row in B]
                assert sa == [[2, 2, 2]] and sb == [[3, 2, 2]]
                wx = max(sum(sa[0]) + sb[0][j] for j in range(3))
                wz = max(sum(sb[0]) + sa[0][j] for j in range(3))
                assert wx == 8 and wz == 9
                assert _grid_weights(sa, sb) == max(wx, wz)
                assert _no_dead_lines(sa, sb)
                if max(wx, wz) > min(9, 12, W_MAX):
                    continue

                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 540 or not (375 * n <= 1000 * k < 425 * n):
                    continue
                key = _h54_key(hx + hz, n)
                if key in published:
                    continue
                if not _h54_quantum4(hz, hx, n):
                    continue
                if not _h54_quantum4(hx, hz, n):
                    continue
                if not css_ok(hx, hz):
                    raise ValueError("verified Heisenberg lift failed CSS")
                tested += 1

                xseed = embedded(bseed, "X", hz, hx)
                zseed = embedded(aseed, "Z", hx, hz)
                xword = _h54_probe(rng, hz, n, 3, xseed, hx)
                zword = _h54_probe(rng, hx, n, 3, zseed, hz)
                if not xword or not zword:
                    continue
                reading = min(xword.bit_count(), zword.bit_count())
                assert reading >= 5
                assert max(row.bit_count() for row in hx + hz) == 9
                quality = (k * reading * reading / n,
                           reading, floor, k, rank)
                entry = (quality, freeze(A), freeze(B), k, reading,
                         floor, (position % 20, position // 20, mode), key)
                if winner is None or quality > winner[0]:
                    winner = entry
                if reading >= 6 or tested >= 3:
                    break

            if winner is not None and winner[4] >= 6:
                break

        if winner is None:
            # No unscreened or equivalent filling is emitted on failure.
            results.append(None)
        else:
            published.add(winner[-1])
            results.append(winner)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure lookup; publication cannot mutate the memoized portfolio."""
    entry = _h54_portfolio(G, N_SLOTS)[idx]
    if entry is None:
        raise ValueError("novel order-six reinforcement card exhausted")
    quality, A, B, k, reading, floor, card, key = entry
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "constituent_B_kernel_floor": floor,
            "quantum_logical_floor": 5,
            "design_card": card, "quality": quality,
            "tanner_invariant": key.hex()}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without sampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: The Frobenius component-screened catalogue consistently gives d30s>4: parent slot 30 fell from d2s=5 to d30s=4, with both X and Z logicals touching both LP sectors.
# REFUTED: Exhaustive Frobenius binomial enumeration supplies five novel k=226 finalists: the current table measured only one fresh k=226 code and four k=222 codes; slots 0 and 1 reproduced archived binomial codes.
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
