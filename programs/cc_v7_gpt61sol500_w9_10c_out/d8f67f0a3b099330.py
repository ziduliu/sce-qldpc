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
CONCEPT = ("Dicyclic Dic5, order 20, A=B=(2,5): diversified cubic incidence and explicit constituent weight-six/seven suppression should supply five fresh measured scores above 15.207 in cell 0.300; distance improvement is a falsifiable measurement hypothesis.")
GROUP = ("Jump to x^10=1, y^2=x^5, y*x*y^-1=x^-1. Dic5 has a unique involution, whereas the parent's C5 semidirect_2 C4 has five, proving nonisomorphism. BEDS includes an order-twenty centre-two/eight-class group, but these statistics do not identify its presentation; this is not claimed to be an untried bed. The parent repeatedly measured [580,180,7], motivating a different group law and incidence search rather than a transferred distance guarantee.")
SHAPES = ("Keep equal (2,5) constituents, each with 100 variables and 40 checks. Factor 29 permits q<=20; q=20 gives n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300. Parent weight-seven minima were entirely sector-1 Acol/Bcol witnesses, prominently Acol1/2 and Bcol2/3, without reported Arow/Brow minima. Verified complete ranks retain [0.275,0.325); useful rank surplus is allowed.")
FILLING = ("Enumerate cubic incidence profiles with column degrees 2,2,2,3,3 and row totals six: BOTH LP weight maxima are nine. Twenty support/relative-order cards optimize scaffold-built constituents without requiring the parent's fixed matching core. Exact syndrome collisions exclude kernels through six; anchored quartet/triple collisions test seven using only constituent regular-action symmetry. Actual seven-word blocks guide bounded single-label repairs. Full lifts enforce rate, CSS, stabilizer-aware logical exclusion through four, and canonical Tanner certificates when available, otherwise conservative intrinsic invariants. Explicit embedded and mixed-sector witnesses rank pairings; lower-floor fallbacks and witness upper bounds are recorded honestly. Unseen archive novelty and GPU distance survival remain unproved.")
FAMILY_NAME = "r030_dic5_q20_25x25_incidence_anchor7_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 requires order twenty")
    # (a,b)=x^a y^b, a mod 10, b in {0,1}.
    # x^10=1, y^2=x^5, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        carry, b = divmod(b, 2)
        return ((a + 5 * carry) % 10, b)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 5 * b * d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a + 5) % 10, 1) if b else ((-a) % 10, 0)

    def idx(g):
        a, b = red(g)
        return 10 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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
    """Extract an arbitrary-shaped constituent using the verified lift."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    checks = len(grid) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:checks]]


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


def _dc_columns(rows, width):
    out = [0] * width
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << i
            row ^= bit
    return out


def _dc_basis(rows):
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


def _dc_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _dc_quantum_four(checks, stabilizers, width):
    """Exact logical exclusion through four, allowing short stabilizers."""
    basis = _dc_basis(stabilizers)
    pivots = sorted(basis, reverse=True)
    residues = []
    for j in range(width):
        word = 1 << j
        for p in pivots:
            if (word >> p) & 1:
                word ^= basis[p]
        residues.append(word)

    cols = _dc_columns(checks, width)
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        residue = residues[j]
        if syndrome in seen and seen[syndrome] != residue:
            return False
        seen.setdefault(syndrome, residue)
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            residue = residues[j] ^ residues[i]
            if syndrome in seen and seen[syndrome] != residue:
                return False
            seen.setdefault(syndrome, residue)
    return True


def _dc_clean_six(rows, width):
    """Exactly exclude every classical kernel support of weight <=6."""
    cols = _dc_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    for j, first in enumerate(cols):
        for i in range(j):
            pair = first ^ cols[i]
            for h in range(i):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _dc_seven(rows, width, q, identity_index):
    """Given no kernels <=6, detect every constituent seven-word orbit.

    A left regular constituent commutes with simultaneous right
    translation of its variable and check coordinates. Every nonempty
    support therefore has a translate containing an identity-coordinate
    variable. Five anchored quartets versus all triples are exhaustive
    for seven-words. This symmetry is NOT assumed for complete LP codes.
    """
    from itertools import combinations
    cols = _dc_columns(rows, width)
    bits = [1 << j for j in range(width)]
    triples = {}
    for a, b, c in combinations(range(width), 3):
        syndrome = cols[a] ^ cols[b] ^ cols[c]
        triples[syndrome] = bits[a] | bits[b] | bits[c]

    for anchor in range(identity_index, width, q):
        others = [j for j in range(width) if j != anchor]
        first = cols[anchor]
        for a, b, c in combinations(others, 3):
            previous = triples.get(first ^ cols[a] ^ cols[b] ^ cols[c])
            if previous is None:
                continue
            word = previous ^ bits[anchor] ^ bits[a] ^ bits[b] ^ bits[c]
            assert word.bit_count() == 7
            assert all((row & word).bit_count() % 2 == 0 for row in rows)
            return word
    return 0


def _dc_probe(rows, width, local, passes, seed=0, own=None):
    """Return an attained kernel/logical witness, never a search cutoff."""
    cols = _dc_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _dc_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
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
                inspect(word)

        local.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            w = word.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            kernel[i] = word
            inspect(word)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])

        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
        if own is None and weight <= 6:
            break
        if own is not None and weight <= 4:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _dc_outside(best, own)
    return best


@lru_cache(maxsize=1)
def _dc_portfolio(G, count):
    """Pure deterministic optimization; only immutable outputs are cached."""
    from itertools import product
    from collections import Counter
    from hashlib import sha256
    try:
        import pynauty
    except ImportError:
        pynauty = None

    q, e, width = G.q, G.e, 5 * G.q
    assert q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    cap = min(9, 12, W_MAX)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def fingerprint(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def order(g):
        power = e
        for length in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid dicyclic element order")

    orders = {g: order(g) for g in G.elements}
    assert sum(orders[g] == 2 for g in G.elements) == 1
    pools = (
        [g for g in G.elements if orders[g] == 4],
        [g for g in G.elements if orders[g] in (5, 10)],
        [g for g in G.elements if orders[g] in (4, 10)],
        [g for g in G.elements if g != e],
    )

    # Remove support row/column permutations before voltage optimization.
    degrees = (2, 2, 2, 3, 3)
    profiles = {}
    for upper in product(*(range(d + 1) for d in degrees)):
        if sum(upper) != 6:
            continue
        lower = tuple(d - a for d, a in zip(degrees, upper))
        key = min(
            tuple(sorted(zip(upper, lower))),
            tuple(sorted(zip(lower, upper))))
        profiles.setdefault(key, (upper, lower))
    profiles = [profiles[key] for key in sorted(profiles)]
    # Interleave the measured-style split support with different incidence.
    split = ((1, 1, 1, 2, 1), (1, 1, 1, 1, 2))
    skey = min(tuple(sorted(zip(*split))),
               tuple(sorted(zip(split[1], split[0]))))
    profiles.sort(key=lambda p: (
        min(tuple(sorted(zip(*p))),
            tuple(sorted(zip(p[1], p[0])))) != skey,
        p))
    assert profiles
    assert all(_no_dead_lines(p, p) for p in profiles)
    assert all(_grid_weights(p, p) == 9 for p in profiles)

    def draw(local, profile, policy):
        grid = [[[] for _ in range(5)] for _ in range(2)]
        for j in range(5):
            for i in range(2):
                terms = profile[i][j]
                if not terms:
                    continue
                anchor = G.pick(local)
                relative = [e]
                while len(relative) < terms:
                    g = local.choice(pools[policy])
                    if g not in relative:
                        relative.append(g)
                grid[i][j] = [G.mul(anchor, g) for g in relative]
            # Fix a column gauge; the remaining coordinates are relative.
            first_row = 0 if grid[0][j] else 1
            inverse = G.inv(grid[first_row][j][0])
            for i in range(2):
                grid[i][j] = [G.mul(g, inverse) for g in grid[i][j]]
                assert len(set(grid[i][j])) == profile[i][j]
        return grid

    def cycle_cost(rows):
        return sum(
            overlap * (overlap - 1) // 2
            for i, row in enumerate(rows) for other in rows[:i]
            for overlap in [(row & other).bit_count()])

    def graph_key(hx, hz, n):
        """Full uncoloured Tanner certificate, or conservative invariant."""
        rows = list(hx) + list(hz)
        if pynauty is None:
            cols = _dc_columns(rows, n)
            value = (
                tuple(sorted(Counter(r.bit_count() for r in rows).items())),
                tuple(sorted(Counter(c.bit_count() for c in cols).items())),
                _tanner_signature(list(hx), list(hz)))
            return sha256(repr(value).encode("ascii")).digest()
        m = len(rows)
        adjacency = {i: [] for i in range(m + n)}
        for i, row in enumerate(rows):
            while row:
                bit = row & -row
                j = m + bit.bit_length() - 1
                adjacency[i].append(j)
                adjacency[j].append(i)
                row ^= bit
        colors = {}
        for i, neighbors in adjacency.items():
            colors.setdefault(len(neighbors), set()).add(i)
        graph = pynauty.Graph(
            number_of_vertices=m + n, directed=False,
            adjacency_dict=adjacency,
            vertex_coloring=[colors[d] for d in sorted(colors)])
        return pynauty.certificate(graph)

    bank, proposed = [], set()
    for card in range(20):
        profile = profiles[(card // 4) % len(profiles)]
        policy = card % 4
        local = random.Random(BASE_SEED ^ (0xD1C530 + 104729 * card))
        clean, fallback = [], []
        for trial in range(96):
            C = draw(local, profile, policy)
            key = fingerprint(C)
            if key in proposed:
                continue
            proposed.add(key)
            rows = _component_rows(G, "A", C)
            rank = gf2_rank(rows)
            if rank < 2 * q - 1:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            cost = cycle_cost(rows)
            record = dict(grid=C, rows=rows, rank=rank, cost=cost,
                          card=card, trial=trial, profile=profile)
            if _dc_clean_six(rows, width):
                clean.append(record)
            else:
                fallback.append(record)

        clean.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))
        fallback.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))
        finalists = []
        for record in clean[:8]:
            word = _dc_seven(record["rows"], width, q, G.idx(e))
            record["floor"] = 7 if word else 8
            record["word"] = word
            finalists.append(record)

        # A bounded repair changes a cell actually used by a seven-word.
        # Both count budgets remain unchanged; no labels are published
        # merely because they belong to another orbit of the same filling.
        if finalists:
            cursor = max(finalists, key=lambda r:
                         (r["floor"], -r["cost"], r["rank"]))
            for repair in range(8):
                if cursor["floor"] != 7:
                    break
                C = [[list(cell) for cell in row]
                     for row in cursor["grid"]]
                touched = [j for j in range(5)
                           if (cursor["word"] >> (j * q)) & ((1 << q) - 1)]
                j = local.choice(touched)
                endpoints = [i for i in range(2) if C[i][j]]
                i = local.choice(endpoints)
                position = local.randrange(len(C[i][j]))
                available = [g for g in G.elements if g not in C[i][j]]
                C[i][j][position] = local.choice(available)
                key = fingerprint(C)
                if key in proposed:
                    continue
                proposed.add(key)
                rows = _component_rows(G, "A", C)
                rank = gf2_rank(rows)
                if rank < 2 * q - 1 or not _dc_clean_six(rows, width):
                    continue
                word = _dc_seven(rows, width, q, G.idx(e))
                record = dict(
                    grid=C, rows=rows, rank=rank, cost=cycle_cost(rows),
                    card=card, trial=96 + repair, profile=profile,
                    floor=7 if word else 8, word=word)
                finalists.append(record)
                if (record["floor"], -record["cost"]) >= (
                        cursor["floor"], -cursor["cost"]):
                    cursor = record

        # Lower-floor fallbacks are explicit, never advertised as clean-six.
        if len(finalists) < 4:
            for record in fallback[:4]:
                record["floor"], record["word"] = 5, 0
                finalists.append(record)
        finalists.sort(key=lambda r:
                       (-r["floor"], r["cost"], -r["rank"], r["trial"]))

        for record in finalists[:4]:
            word = _dc_probe(record["rows"], width, local, 16,
                             record["word"])
            if not word:
                continue
            assert word.bit_count() >= record["floor"]
            right = _component_rows(G, "B", record["grid"])
            transferred, source = 0, word
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                transferred |= 1 << (
                    j * q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            assert all((row & transferred).bit_count() % 2 == 0
                       for row in right)
            transferred = _dc_probe(right, width, local, 16, transferred)
            record["left"], record["right"] = word, transferred
            record["reading"] = min(
                word.bit_count(), transferred.bit_count())
            record["grid"] = freeze(record["grid"])
            bank.append(record)

    if len(bank) < 5:
        raise ValueError("too few dicyclic incidence constituents")
    bank.sort(key=lambda r:
              (-r["reading"], -r["floor"], r["cost"],
               r["card"], r["trial"]))

    def descend(word, stabilizers):
        if not word:
            return 0
        for _ in range(3):
            previous = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            starting = word
            moves = sorted(
                (row for row in stabilizers
                 if (row & starting).bit_count() >= 2),
                key=lambda row: -(row & starting).bit_count())[:48]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == previous:
                break
        return word

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(5):
            if kind == "X":
                word = record["right"] << (block * width)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 5 + block) * q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _dc_outside(word, own):
                continue
            word = descend(word, stabilizers)
            assert word and _dc_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -min(a["floor"], b["floor"]),
         a["cost"] + b["cost"],
         int(a["card"] == b["card"]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    graph_keys, results = set(), []
    tested = 0
    for _, _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert wx == wz == 9 and max(wx, wz) <= cap
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275 * n <= 1000 * k < 325 * n):
            continue
        key = graph_key(hx, hz, n)
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1
        if not _dc_quantum_four(hz, hx, n):
            continue
        if not _dc_quantum_four(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap

        bx, bz = _dc_basis(hx), _dc_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xD1C5E0 + 8191 * ai + bi))
        xword = _dc_probe(hz, n, local, 3, xseed, bx)
        zword = _dc_probe(hx, n, local, 3, zseed, bz)
        xword, zword = descend(xword, hx), descend(zword, hz)
        assert _dc_outside(xword, bx) and _dc_outside(zword, bz)
        assert all((row & xword).bit_count() % 2 == 0 for row in hz)
        assert all((row & zword).bit_count() % 2 == 0 for row in hx)
        reading = min(xword.bit_count(), zword.bit_count())
        quality = (k * reading * reading / n, reading, k,
                   min(a["floor"], b["floor"]),
                   -a["cost"] - b["cost"])
        results.append((
            quality, A, B, k, reading,
            (a["card"], b["card"]),
            (a["floor"], b["floor"]),
            (a["reading"], b["reading"])))
        if len(results) >= count + max(12, count):
            break
        if tested >= max(1024, 12 * count) and len(results) >= count:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one screened dicyclic prescription without further draws."""
    catalogue = _dc_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct dicyclic incidence portfolio exhausted")
    quality, A, B, k, reading, cards, floors, components = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "constituent_kernel_floors": floors,
            "constituent_witness_upper_bounds": components,
            "design_cards": cards, "quality": quality,
            "certified_logical_floor": 5}


from functools import lru_cache


@lru_cache(maxsize=1)
def _cubic_core_portfolio(G):
    """Memoization changes cost, not the deterministic published cells.

    All matrix screening uses lifted_matrices. Unequal graph invariants
    certify within-program inequivalence; unseen archive novelty remains
    the evaluator's responsibility.
    """
    from collections import Counter
    from hashlib import sha256

    assert G.q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    local = random.Random(BASE_SEED ^ 0x205C081)
    q, width, cap = G.q, 5 * G.q, min(9, W_MAX)
    if cap < 9:
        raise ValueError("cubic-core profiles require row cap nine")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

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
        out = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def outside(word, span):
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def clean_six(rows, n):
        """Exactly exclude all nonzero constituent words of weight <=6.

        Every such support splits into two supports of size at most
        three. Equal syndromes of distinct supports detect their XOR.
        """
        cols = columns(rows, n)
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return False
            seen.add(syndrome)
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        for j, first in enumerate(cols):
            for i in range(j):
                pair = first ^ cols[i]
                for h in range(i):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return False
                    seen.add(syndrome)
        return True

    def quantum_four(checks, own, n):
        """Exact logical exclusion through four; short stabilizers are legal."""
        span = basis(own)
        pivots = sorted(span, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in pivots:
                if (word >> p) & 1:
                    word ^= span[p]
            residues.append(word)

        cols = columns(checks, n)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            residue = residues[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = residue
            elif old != residue:
                return False
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = residues[j] ^ residues[i]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = residue
                elif old != residue:
                    return False
        return True

    def probe(rows, n, rounds, own=None):
        """Return an actual kernel/logical witness weight, or zero.

        An unresolved probe receives no optimistic sentinel. Every
        inspected support is constructed in the verified matrix kernel.
        """
        cols = columns(rows, n)
        span = basis(own) if own is not None else None
        order = list(range(n))
        best = 0

        def inspect(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if best and weight >= best:
                return
            if span is not None and not outside(word, span):
                return
            best = weight

        for _ in range(rounds):
            local.shuffle(order)
            pivots, nulls = {}, []
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
                    nulls.append(word)
                    inspect(word)

            short = sorted(nulls, key=int.bit_count)[:36]
            for word in nulls:
                for other in short:
                    candidate = word ^ other
                    if 0 < candidate.bit_count() < word.bit_count():
                        word = candidate
                inspect(word)
            for j, first in enumerate(short):
                for i in range(j):
                    pair = first ^ short[i]
                    inspect(pair)
                    for h in range(i):
                        inspect(pair ^ short[h])
            if best and best <= 6:
                break
        return best

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured full-Tanner invariant, not label data."""
        rows = list(hx) + list(hz)
        value = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count()
                                 for c in columns(rows, n)).items())),
            _tanner_signature(list(hx), list(hz)),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def core_info(g, h):
        core = [[[G.e], [G.e], [G.e]],
                [[G.e], [g], [h]]]
        rows = _component_rows(G, "B", core)
        cols = columns(rows, 3 * q)
        adj = [[] for _ in rows]
        for edge, syndrome in enumerate(cols):
            if syndrome.bit_count() != 2:
                return None
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            adj[a].append((b, edge))
            adj[b].append((a, edge))
        girth = len(rows) + 1
        for root in range(len(rows)):
            distance = [-1] * len(rows)
            parent_edge = [-1] * len(rows)
            distance[root] = 0
            queue = [root]
            for a in queue:
                for b, edge in adj[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a] + 1
                        parent_edge[b] = edge
                        queue.append(b)
                    elif edge != parent_edge[a] and edge != parent_edge[b]:
                        girth = min(girth, distance[a] + distance[b] + 1)
        return girth

    cores = []
    for g in G.elements:
        if g == G.e:
            continue
        for h in G.elements:
            if h == G.e or h == g:
                continue
            girth = core_info(g, h)
            if girth is not None and girth >= 8:
                cores.append((girth, g, h))
    if not cores:
        raise ValueError("no verified matching core of girth eight")
    best_girth = max(item[0] for item in cores)
    cores = [item for item in cores if item[0] == best_girth]

    def element_order(g):
        power = G.e
        for t in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return t
        raise ValueError("invalid group order")

    rotations = [g for g in G.elements if element_order(g) == 5]
    quartics = [g for g in G.elements if element_order(g) == 4]
    long = rotations + quartics

    def draw(profile, policy):
        _, g, h = local.choice(cores)
        C = [[[] for _ in range(5)] for _ in range(2)]
        mono = (2, 3, 4) if profile == 0 else (1, 2, 3)
        for j, relative in zip(mono, (G.e, g, h)):
            C[0][j], C[1][j] = [G.e], [relative]

        pool = rotations if policy == 0 else quartics if policy == 1 else long
        if profile == 0:
            u, v = local.choice(pool), local.choice(pool)
            p, t = G.pick(local), G.pick(local)
            C[0][0], C[1][0] = [G.e, u], [p]
            C[0][1], C[1][1] = [G.e], [t, G.mul(t, v)]
        else:
            for row, j in ((0, 0), (1, 4)):
                for _ in range(64):
                    u, v = local.choice(pool), local.choice(pool)
                    if u == v:
                        continue
                    commute = G.mul(u, v) == G.mul(v, u)
                    if policy == 3 and commute:
                        continue
                    if policy == 4 and not commute:
                        continue
                    C[row][j] = [G.e, u, v]
                    break
                if not C[row][j]:
                    return None
        return C

    # Build new constituent matrices, not perturbations of archived cells.
    bank, fill_keys, component_keys = [], set(), set()
    for attempt in range(8192):
        profile, policy = attempt % 2, (attempt // 2) % 5
        C = draw(profile, policy)
        if C is None:
            continue
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in C for cell in row)
        if filling in fill_keys:
            continue
        fill_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        assert all(sum(row) == 6 for row in counts)
        assert max(sum(row[j] for row in counts) for j in range(5)) == 3
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) == 9

        left = _component_rows(G, "A", C)
        right = _component_rows(G, "B", C)
        if gf2_rank(left) != 2 * q or gf2_rank(right) != 2 * q:
            continue
        if _kernel_profile(left, width)[0] != 5:
            continue
        if not clean_six(left, width) or not clean_six(right, width):
            continue
        signature = _tanner_signature(left, [])
        if signature in component_keys:
            continue
        component_keys.add(signature)
        dl, dr = probe(left, width, 10), probe(right, width, 10)
        if not dl or not dr:
            continue
        assert min(dl, dr) >= 7
        bank.append((min(dl, dr), profile, policy, freeze(C), dl, dr))
        if len(bank) >= 64:
            break
    if len(bank) < 8:
        raise ValueError("too few distinct kernel-six-clean constituents")
    bank.sort(key=lambda item: (-item[0], item[1], item[2],
                                repr(item[3])))

    # Reprobe stronger candidates with the same extra budget.
    refined = []
    for entry in bank[:40]:
        _, profile, policy, C, dl, dr = entry
        dl = min(dl, probe(_component_rows(G, "A", C), width, 24))
        dr = min(dr, probe(_component_rows(G, "B", C), width, 24))
        refined.append((min(dl, dr), profile, policy, C, dl, dr))
    bank = sorted(refined, key=lambda item:
                  (-item[0], item[1], item[2], repr(item[3])))

    excluded = set()
    # The reported F20 frontier filling may use action 2 or its inverse.
    reference = [
        [[(0, 1), (2, 1)], [(3, 2)], [G.e], [G.e], [G.e]],
        [[(0, 2)], [G.e, (3, 1)], [G.e], [(2, 0)], [(1, 3)]],
    ]
    for sign in (1, -1):
        C = [[[(a, sign * b % 4) for a, b in cell]
              for cell in row] for row in reference]
        hx, hz, n = lifted_matrices(G, C, C)
        excluded.add(graph_key(hx, hz, n))

    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    tested = set()
    entries = []
    for position in range(N_SLOTS):
        card, epoch = position % 20, position // 20
        support_pair = ((1, 1), (0, 1), (0, 0))[card % 3]
        pressure = 1 + (card // 3 + epoch) % 3
        ranked = list(plans)
        local.shuffle(ranked)
        ranked.sort(key=lambda ab: (
            tuple(sorted((bank[ab[0]][1], bank[ab[1]][1]))) != support_pair,
            -(pressure * bank[ab[0]][0]
              + (4 - pressure) * bank[ab[1]][0]),
            bank[ab[0]][2] == bank[ab[1]][2],
        ))
        survivors = []
        for a, b in ranked:
            if (a, b) in tested:
                continue
            tested.add((a, b))
            A, B = bank[a][3], bank[b][3]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 580 or k != 180:
                continue
            if not (275 * n <= 1000 * k < 325 * n):
                continue
            signature = graph_key(hx, hz, n)
            if signature in excluded:
                continue
            if not quantum_four(hz, hx, n):
                continue
            if not quantum_four(hx, hz, n):
                continue

            # Reserve even discarded survivors, preventing slot recycling.
            excluded.add(signature)
            dx = probe(hz, n, 3, hx)
            dz = probe(hx, n, 3, hz)
            reading = min(dx, dz) if dx and dz else 5
            quality = (
                k * reading * reading / float(n),
                min(bank[a][0], bank[b][0]),
                -(pressure * bank[a][4] + (4 - pressure) * bank[b][5]),
            )
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            survivors.append((quality, A, B, k, reading,
                              (card, epoch, bank[a][1:3], bank[b][1:3])))
            if len(survivors) >= 3:
                break
        if not survivors:
            raise ValueError("invariant-distinct cubic-core portfolio exhausted")
        winner = max(survivors, key=lambda item: item[0])
        _, A, B, k, reading, card_info = winner
        entries.append((A, B, k, reading, card_info))
    return tuple(entries)


def _inactive_order55_catalogue(rng, G, idx):
    """Historical reference only; NEW_CTX never calls this old catalogue."""
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


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: The Frobenius component-screened catalogue consistently gives d30s>4: parent slot 30 fell from d2s=5 to d30s=4, with both X and Z logicals touching both LP sectors.
# REFUTED: Exhaustive canonical binomial selection supplies five fresh k=226 finalists: the current evaluation measured one fresh k=226 code and four k=222 codes, while slots 0 and 1 reproduced archived codes.
# REFUTED: The order-20 affine fixed-core cubic portfolio supplies five fresh distance-eight readings: all five current measured finalists had k=180,d30s=7, and sixty slots retained only six distinct codes.
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
