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
CONCEPT = ("C11 semidirect C4 with inversion action, order 44, A=B=(2,3): independently optimized rank-deficient saturated constituents should supply five fresh finalists with measured fitness above 46.5234 in cell 0.100; retaining distance-24 readings while raising k toward 52 is the falsifiable goal.")
GROUP = ("Keep the metacyclic semidirect-product construction rule, changing parameters to m=11,r=4,s=10 because factor 13 makes order 55 inadmissible. Order 44 gives n=572. This group is isomorphic to Dic11: its normal C11 and order-four complement reproduce the abstract group of the measured 46.5234 bed. That measured success motivates this move but does not certify distance or archive novelty.")
SHAPES = ("Change both sides to (2,3), giving equal 132-variable, 88-check constituents, n=572, kb=44 and guaranteed rate 1/13. The size window permits q<=46 and q=44 fits. Redirect to cell 0.100: actual k around 52 still lies below rate 0.125. Parent weight-five minima occupied sector-1 Acol/Bcol blocks broadly, with some X families concentrated at Bcol2; independent optimization strengthens both constituent mechanisms. Actual full ranks, not constituent-rank predictions, determine acceptance.")
FILLING = ("Saturated support cards give every constituent row five terms and every column at most four, so BOTH CSS maxima are at most nine. Prefer augmentation-rank-one profiles and constituent rank deficit two, then screen exact normal-C11 invariant words and equally refine explicit kernel witnesses. Pair independently selected constituents, enforce the actual low-rate band, exclude complete kernel supports through four, and reduce verified nontrivial logical witnesses by stabilizer pairs. Different intrinsic uncoloured Tanner signatures separate slots conservatively. Rank surplus is rewarded only alongside equally searched witness readings; all larger readings remain upper bounds.")
FAMILY_NAME = "r010_m11r4s10_23x23_rank2_saturated_witness"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Same metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Inversion has order two, dividing r=4. This group is Dic11.
    m, r, s = 11, 4, 10
    if not valid_q(q):
        raise ValueError("C11 semidirect C4 has order 44")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

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

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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
    """Extract either rectangular constituent using the verified lift."""
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


def _rank_saturated_design(rng, G, idx, excluded):
    """Optimize one new shape card; every screen uses verified matrices."""
    from hashlib import sha256

    q, width = G.q, 3 * G.q
    assert q == 44
    profiles = (
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
        ((1, 2, 2), (2, 1, 2)),
        ((1, 1, 3), (2, 2, 1)),
    )
    # The first two profiles have augmentation rank one. The other
    # two provide legal fallbacks, not assumed rank-surplus successes.
    pa = (idx // 3) % 2
    pb = (idx // 6 + idx % 2) % 2
    policy = idx % 3
    epoch = idx // 12

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def draw(profile, side):
        mixed = set()
        for j in range(3):
            doubles = [(i, j) for i in range(2) if profile[i][j] == 2]
            if policy == 0:
                mixed.update(doubles)
            elif doubles:
                mixed.add(doubles[(j + epoch + int(side == "B"))
                                  % len(doubles)])
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                elif count == 2:
                    if (i, j) in mixed:
                        relative = (rng.randrange(11), rng.choice((1, 3)))
                    else:
                        relative = (rng.randrange(1, 11),
                                    rng.choice((0, 2)))
                    cell = [first, G.mul(first, relative)]
                else:
                    rotation = (rng.randrange(1, 11),
                                rng.choice((0, 2)))
                    cross = (rng.randrange(11), rng.choice((1, 3)))
                    if (i + j + policy + epoch + int(side == "B")) % 2:
                        second = G.mul(first, cross)
                        cell = [first, second, G.mul(second, rotation)]
                    else:
                        cell = [first, G.mul(first, rotation),
                                G.mul(first, cross)]
                assert len({G._key(g) for g in cell}) == count
                row.append(cell)
            grid.append(row)
        return grid

    def invariant_word(cols):
        """Exact minimum within the complete normal-C11 invariant space.

        Twelve full cosets, not a hand-written constituent lift.
        This restricted-space minimum is an actual kernel upper bound.
        """
        words, syndromes = [], []
        for j in range(3):
            for b in range(4):
                word = syndrome = 0
                for a in range(11):
                    pos = j * q + G.idx((a, b))
                    word |= 1 << pos
                    syndrome ^= cols[pos]
                words.append(word)
                syndromes.append(syndrome)
        previous = word = syndrome = 0
        best = 0
        for number in range(1, 1 << len(words)):
            gray = number ^ (number >> 1)
            changed = (gray ^ previous).bit_length() - 1
            word ^= words[changed]
            syndrome ^= syndromes[changed]
            if not syndrome and (not best
                                 or word.bit_count() < best.bit_count()):
                best = word
            previous = gray
        return best

    def probe(rows, passes, seed):
        """Explicit constituent kernel witnesses; never a distance floor."""
        cols = columns(rows, width)
        best = seed

        def inspect(word):
            nonlocal best
            if word and (not best or word.bit_count() < best.bit_count()):
                best = word

        for _ in range(passes):
            order = list(range(width))
            rng.shuffle(order)
            pivots, kernel = {}, []
            for j in order:
                syn, word = cols[j], 1 << j
                while syn:
                    p = syn.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syn, word)
                        break
                    syn ^= old[0]
                    word ^= old[1]
                if not syn:
                    kernel.append(word)
                    inspect(word)

            # Elementary exchanges preserve the complete kernel span.
            rng.shuffle(kernel)
            for i, word in enumerate(kernel):
                for other in kernel:
                    trial = word ^ other
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
                kernel[i] = word
                inspect(word)
            for i, first in enumerate(kernel):
                for second in kernel[:i]:
                    inspect(first ^ second)

            ranked = sorted(kernel, key=int.bit_count)
            short = ranked[:24]
            tail = ranked[24:]
            short += rng.sample(tail, min(8, len(tail)))
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    for h in range(j):
                        inspect(pair ^ short[h])
            for _ in range(2):
                starting = best
                for i, first in enumerate(short):
                    inspect(starting ^ first)
                    for second in short[:i]:
                        inspect(starting ^ first ^ second)
                if best == starting:
                    break
        assert best
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return best

    def finalists(side, primary, batch):
        pool = []
        for trial in range(192):
            p = primary
            if batch and trial % 4 == 0:
                p = 2 + (trial // 4 + idx) % 2
            grid = draw(profiles[p], side)
            rows = _component_rows(G, side, grid)
            rank = gf2_rank(rows)
            reading, collisions = _kernel_profile(rows, width)
            if reading != 5:
                continue
            cols = columns(rows, width)
            seed = invariant_word(cols)
            if not seed:
                continue
            c4 = 0
            for i, first in enumerate(cols):
                for second in cols[:i]:
                    overlap = (first & second).bit_count()
                    c4 += overlap * (overlap - 1) // 2
            deficit = 2 * q - rank
            cheap = (
                int(seed.bit_count() <= 22),
                abs(deficit - 2), c4, collisions, trial)
            pool.append({
                "grid": grid, "rows": rows, "seed": seed,
                "rank": rank, "deficit": deficit, "cheap": cheap,
            })
        pool.sort(key=lambda item: item["cheap"])
        shortlist = pool[:8]
        for record in shortlist:
            record["word"] = probe(record["rows"], 4, record["seed"])
            record["reading"] = record["word"].bit_count()
        shortlist.sort(key=lambda item: (
            item["reading"],
            -abs(item["deficit"] - 2),
            -item["cheap"][2]), reverse=True)
        out = shortlist[:3]
        # Every constituent eligible for emission gets the same 20 passes.
        for record in out:
            record["word"] = probe(record["rows"], 16, record["word"])
            record["reading"] = record["word"].bit_count()
        return out

    def signature(hx, hz, n):
        value = (
            n, tuple(sorted(row.bit_count() for row in hx + hz)),
            tuple(sorted(c.bit_count() for c in columns(hx + hz, n))),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def quantum_reading(Arecord, Brecord, hx, hz, n):
        """Embed, verify and reduce actual nontrivial logical witnesses."""
        best = None
        for kind, checks, own in (("X", hz, hx), ("Z", hx, hz)):
            span = {}
            for row in own:
                while row:
                    p = row.bit_length() - 1
                    if p in span:
                        row ^= span[p]
                    else:
                        span[p] = row
                        break

            def nontrivial(word):
                while word:
                    p = word.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        return True
                    word ^= old
                return False

            for block in range(3):
                if kind == "X":
                    word = Brecord["word"] << (block * width)
                else:
                    source, word = Arecord["word"], 0
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 3 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not nontrivial(word):
                    continue
                weight = word.bit_count()
                for _ in range(2):
                    starting = word
                    for i, first in enumerate(own):
                        trial = starting ^ first
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                        for second in own[:i]:
                            trial = starting ^ first ^ second
                            if trial.bit_count() < weight:
                                word, weight = trial, trial.bit_count()
                    if word == starting:
                        break
                assert word and nontrivial(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = weight if best is None else min(best, weight)
        return best

    winner = None
    winner_quality = None
    for batch in range(2):
        aa = finalists("A", pa, batch)
        bb = finalists("B", pb, batch)
        pending = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                sa = [[len(cell) for cell in row] for row in A]
                sb = [[len(cell) for cell in row] for row in B]
                assert _no_dead_lines(sa, sb)
                assert _grid_weights(sa, sb) <= min(9, W_MAX)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 572 or not (k > 0 and 8 * k < n):
                    continue
                weak = min(a["reading"], b["reading"])
                priority = (
                    k * weak * weak / float(n), weak, k,
                    -abs(a["deficit"] - 2) - abs(b["deficit"] - 2))
                pending.append((priority, a, b, hx, hz, n, k))
        pending.sort(key=lambda item: item[0], reverse=True)

        tested = 0
        for _, a, b, hx, hz, n, k in pending:
            sig = signature(hx, hz, n)
            if sig in excluded:
                continue
            # Complete kernels include mixed-sector words. This stronger
            # gate also rejects short stabilizers rather than rewarding
            # an unresolved component-only diagnostic.
            if _kernel_profile(hx, n)[0] != 5:
                continue
            if _kernel_profile(hz, n)[0] != 5:
                continue
            assert css_ok(hx, hz)
            reading = quantum_reading(a, b, hx, hz, n)
            if reading is None:
                continue
            quality = (k * reading * reading / float(n), reading, k)
            if winner_quality is None or quality > winner_quality:
                assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
                winner_quality = quality
                winner = (
                    a["grid"], b["grid"], k, sig, reading,
                    (a["reading"], b["reading"]),
                )
            tested += 1
            if tested >= 3:
                break
        if winner is not None and winner_quality[1] >= 24:
            break
    if winner is None:
        raise ValueError("rank-surplus saturated card exhausted")
    return winner


from functools import lru_cache


@lru_cache(maxsize=None)
def _rank_saturated_prefix(G, count):
    """Immutable deterministic prefix; memoization changes no result."""
    if count == 0:
        return ()
    previous = _rank_saturated_prefix(G, count - 1)
    excluded = {entry[3] for entry in previous}
    idx = count - 1
    winner = None
    for restart in range(3):
        local = random.Random(
            BASE_SEED * 10007 + 0x442352
            + 65537 * idx + 1000003 * restart)
        try:
            winner = _rank_saturated_design(local, G, idx, excluded)
            break
        except ValueError:
            continue
    if winner is None:
        raise ValueError("invariant-distinct saturated prefix exhausted")
    A, B, k, sig, reading, components = winner

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    entry = (freeze(A), freeze(B), k, sig, reading, components)
    return previous + (entry,)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a fresh, screened design for the changed shapes."""
    A, B, k, sig, reading, components = _rank_saturated_prefix(G, idx + 1)[-1]
    return {
        "A": A, "B": B, "screen_k": k,
        "quantum_witness_upper_bound": reading,
        "component_witness_upper_bounds": components,
        "tanner_signature": sig.hex(),
    }


def _legacy_order55_catalogue(rng, G, idx):
    """Inactive order-55 reference; never called by the new NEW_CTX."""
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
# REFUTED: Exhaustive connected binomial enumeration supplies five novel k=226 finalists: the current table measured only one fresh k=226 code and four k=222 codes; slots 0 and 1 reproduced archived binomial codes.
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
