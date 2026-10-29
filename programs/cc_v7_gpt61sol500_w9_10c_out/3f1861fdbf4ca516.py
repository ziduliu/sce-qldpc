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
CONCEPT = ("On (C3 x C3) semidirect C4 with faithful quarter-turn action, order 36, A=(2,3), B=(2,4), independently optimized saturated constituents should produce five novel cell-0.150 finalists with measured fitness above 25.2. Larger witness readings are upper bounds, not distance certificates.")
GROUP = ("Jump from cyclic Z13 to the Frobenius group (C3 x C3):C4. The quarter-turn has square -I and no nonzero fixed translation. This order-36 bed is absent from BEDS; related order-36 (2,3)x(2,4) beds reached 24.3, versus the cell-0.150 record 25.2. These motivate a new bed, not a transferred distance guarantee.")
SHAPES = ("A=(2,3), B=(2,4) give 108-variable/72-check and 144-variable/72-check constituents, n=576, kb=72 and guaranteed rate 0.125. Factor sixteen permits q<=37; this group fixes q=36. Actual ranks must give 0.125<k/n<0.175, placing publication in cell 0.150. Parent minima occupied sector-1 Acol/Bcol blocks, especially Acol1/2, with no reported sector-2 minima.")
FILLING = ("Cross saturated support profiles with four relative-coset policies. Legal pairs have A-row plus B-column and B-row plus A-column maxima <=9, also below twelve. Independently optimize verified constituents using exact kernel-four rejection, complete normal-C3xC3 invariant-subspace enumeration and equally budgeted explicit-witness searches. Complete verified lifts enforce actual target-band ranks, CSS and kernel-four rejection; nontrivial embedded logicals undergo stabilizer-pair descent and full-kernel probing. A deterministic immutable portfolio conservatively rejects equal full-Tanner invariants. Distance improvement and unseen archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r015_f9_c4_quarterturn_q36_23x24_independent_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("(C3 x C3):C4 requires order 36")
    # (a,b,t) denotes translation (a,b) followed by y^t.
    # T(a,b)=(-b,a), T^2=-I and T^4=I over F3.
    els = [(a, b, t) for t in range(4)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3, int(g[2]) % 4)

    def action(a, b, t):
        return ((a, b), (-b, a), (-a, -b), (b, -a))[t % 4]

    def mul(g, h):
        a, b, t = red(g)
        c, d, u = red(h)
        c, d = action(c, d, t)
        return ((a + c) % 3, (b + d) % 3, (t + u) % 4)

    def inv(g):
        a, b, t = red(g)
        ti = (-t) % 4
        a, b = action(a, b, ti)
        return ((-a) % 3, (-b) % 3, ti)

    def idx(g):
        a, b, t = red(g)
        return 9*t + 3*b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _syndrome_columns(rows, n):
    """Column syndromes of a matrix returned by the verified lift."""
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _short_pairs(cols):
    """Return pair syndromes, or None if a kernel support of size 1..4 exists.

    With distinct nonzero columns, a collision between different
    unordered pairs necessarily has four distinct endpoints.
    """
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = {}
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles or s in pairs:
                return None
            pairs[s] = (j, i)
    return pairs


def _has_five(cols, pairs, start=2):
    """Exact weight-five existence test after weights 1..4 are excluded.

    Restricting the largest triple index to sector 2 tests every mixed
    weight-five support: each such support has a triple partition
    containing a sector-2 endpoint. Overlapping matches would give a
    smaller kernel word, already excluded by _short_pairs.
    """
    for i in range(max(2, start), len(cols)):
        x = cols[i]
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                if (pair ^ cols[h]) in pairs:
                    return True
    return False


def _six_count(cols):
    """Exact weight-six multiplicity, assuming no kernel words <=5."""
    seen = {}
    collisions = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                old = seen.get(syndrome, 0)
                collisions += old
                seen[syndrome] = old + 1
    # Smaller words excluded: matching triples are disjoint, and each
    # six-support has ten unordered partitions into two triples.
    return collisions // 10


def _chain_component(G, idx):
    """Row totals 6,6,6; endpoint trinomials and six degree-two edges."""
    q = G.q

    def e(a):
        return G.elements[a % q]

    C = [[[] for _ in range(8)] for _ in range(3)]
    upper = 3 + idx % 4
    lower = 3 + (idx // 4) % 4
    scale = 1 + (idx // 16) % 12
    C[0][0] = [e(0), e(1), e(upper)]
    C[2][1] = [e(0), e(scale), e(scale * lower)]

    # The two directed-difference sets partition Z13's nonzero elements:
    # {0,1,4} gives +/-{1,3,4}; {0,2,7} gives +/-{2,5,6}.
    # This avoids four-cycles in the degree-two chain.
    for j, voltage in enumerate((0, 1, 4), 2):
        C[0][j] = [e(0)]
        C[1][j] = [e(voltage)]
    for j, voltage in enumerate((0, 2, 7), 5):
        C[1][j] = [e(0)]
        C[2][j] = [e(voltage)]
    return C


def _crossed_component(G, gap0, gap1, h0, h1, third):
    """Two crossed binomials; row totals six, column totals 3,3,2,2,2."""
    q = G.q

    def e(a):
        return G.elements[a % q]

    A = [
        [[e(0)] for _ in range(5)],
        [[e(a)] for a in (h0, h1, 0, 1, third)],
    ]
    A[0][0] = [e(0), e(gap0)]
    A[1][1] = [e(h1), e(h1 + gap1)]
    return A


def _components_from_verified(hx, hz, q):
    """Extract both constituents using the evaluator's actual conventions."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            compact = 0
            for j in range(ca):
                compact |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _inactive_cyclic_context(rng, G, idx):
    """Historical cyclic context; the new entry point never calls this."""
    if G.q != 13:
        raise ValueError("chain/crossed bed requires order thirteen")
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    B = _chain_component(G, idx)

    sa = ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1))
    sb = tuple(tuple(len(cell) for cell in row) for row in B)
    if not _no_dead_lines(sa, sb):
        raise ValueError("dead line in chain/crossed support")
    if _grid_weights(sa, sb) > min(9, W_MAX):
        raise ValueError("chain/crossed support exceeds strict cap")

    # Screen B once, using a verified lift with a shape-compatible dummy A.
    dummy = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dhx, dhz, _ = lifted_matrices(G, dummy, B)
    _, br = _components_from_verified(dhx, dhz, q)
    bc = _syndrome_columns(br, cb * q)
    bp = _short_pairs(bc)
    if bp is None or _has_five(bc, bp):
        raise ValueError("indexed B chain failed exact kernel-five screen")
    b6 = _six_count(bc)
    del bp, bc, dhx, dhz

    # Cards change relative gaps, not a common automorphism or gauge.
    gap0 = 1 + idx % 6
    gap1 = 1 + (idx // 6) % 6
    target_a6 = q * (1 + idx % 10 + 2 * (idx // 20))
    plans = [(h0, h1, third)
             for third in (3, 4, 5, 6)
             for h0 in range(q) for h1 in range(q)]
    rng.shuffle(plans)
    pool = []

    for trial, (h0, h1, third) in enumerate(plans):
        A = _crossed_component(G, gap0, gap1, h0, h1, third)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, _ = _components_from_verified(hx, hz, q)
        ac = _syndrome_columns(ar, ca * q)
        ap = _short_pairs(ac)
        if ap is None or _has_five(ac, ap):
            continue
        a6 = _six_count(ac)
        objective = (abs(a6 - target_a6), a6, trial)
        pool.append((objective, A, hx, hz, n, a6))
        if len(pool) >= 32:
            break

    pool.sort(key=lambda item: item[0])
    n1 = ca * cb * q
    for objective, A, hx, hz, n, a6 in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.325 * n <= k < 0.375 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
            continue

        passed = True
        for checks in (hx, hz):
            cols = _syndrome_columns(checks, n)
            pairs = _short_pairs(cols)
            if pairs is None:
                passed = False
                break

            # A pure sector-1 kernel splits into independent constituent
            # kernels, already screened through five above. Therefore
            # only triples containing sector 2 remain to be tested.
            if _has_five(cols, pairs, n1):
                passed = False
                break
            del pairs, cols
        if not passed:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified lift failed CSS commutation")
        return {"A": A, "B": B, "idx": idx,
                "screen": (k, a6, b6),
                "card": (gap0, gap1, target_a6)}

    raise ValueError("no target-rate filling passed exact kernel-five screens")


from functools import lru_cache


def _r15_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _r15_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        previous = pivots.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _r15_component(G, grid, side):
    """Only scaffold-built actions are screened; zero dummy is not emitted."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows], width


def _r15_invariant_word(G, rows, width):
    """Shortest word in the COMPLETE normal-translation invariant subspace.

    Each orbit is one nine-element translation coset in one variable
    block. Elimination and enumeration use actual verified syndromes.
    This is exact for this subspace, not for the whole constituent.
    """
    cols = _syndrome_columns(rows, width)
    pivots, kernel = {}, []
    for block in range(width // G.q):
        for t in range(4):
            word = syndrome = 0
            for b in range(3):
                for a in range(3):
                    j = block * G.q + G.idx((a, b, t))
                    word |= 1 << j
                    syndrome ^= cols[j]
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)

    best = word = previous_gray = 0
    for number in range(1, 1 << len(kernel)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous_gray).bit_length() - 1
        word ^= kernel[changed]
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word
        previous_gray = gray
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _r15_cycle_cost(rows):
    """Actual Tanner four-cycle count, used only as a cheap heuristic."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _r15_probe(rows, width, rng, passes, seed=0, own=None):
    """Return an explicit kernel/logical witness; never a lower bound."""
    cols = _syndrome_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _r15_outside(word, own):
            return
        best, weight = word, word.bit_count()

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        # Elementary exchanges preserve the complete kernel span.
        rng.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            for j in range(len(kernel)):
                if i == j:
                    continue
                trial = word ^ kernel[j]
                if trial and trial.bit_count() < word.bit_count():
                    word = trial
            kernel[i] = word
            inspect(word)

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])

        for descent in range(2):
            if not best:
                break
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
        if weight <= 5:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _r15_outside(best, own)
    return best


def _r15_key(rows, width):
    """Conservative full-Tanner invariant, pooled without X/Z colours.

    Intersection multiplicities, common-neighbour profiles and incidence
    refinement are permutation invariant. Equal keys are discarded;
    equality is not claimed to prove isomorphism or archive novelty.
    """
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    degrees = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            c = (row & rows[j]).bit_count()
            if c:
                overlaps[(i, j)] = c
                relation = one if c == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(degrees[j],) + feature] += 1
            profiles[j][(degrees[i],) + feature] += 1
    features = [
        (degrees[i], tuple(sorted(profiles[i].items())))
        for i in range(m)]
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]

    incidence = []
    for syndrome in _syndrome_columns(rows, width):
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        incidence.append(neighbors)

    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in neighbors))
                     for neighbors in incidence]
        qpalette = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        following = [[] for _ in rows]
        for neighbors, feature in zip(incidence, qfeatures):
            for r in neighbors:
                following[r].append(qpalette[feature])
        refined = [(colors[r], tuple(sorted(following[r])))
                   for r in range(m)]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return sha256(repr((width, m, history)).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _r15_portfolio(G, count):
    """Deterministic independent optimization; cached outputs are immutable."""
    if G.q != 36:
        raise ValueError("quarter-turn portfolio requires order 36")
    q, e = G.q, G.e

    aprofiles = (
        ((1, 2, 2), (2, 1, 2)),
        ((1, 1, 3), (2, 2, 1)),
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
        ((1, 1, 2), (2, 2, 1)),
        ((1, 1, 2), (1, 2, 1)),
    )
    bprofiles = (
        ((2, 1, 1, 1), (1, 2, 1, 1)),
        ((1, 1, 3, 0), (1, 1, 0, 3)),
        ((1, 1, 1, 2), (1, 1, 1, 2)),
        ((1, 2, 2, 0), (1, 2, 0, 2)),
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, 1, 1, 3), (1, 2, 2, 1)),
        ((1, 1, 2, 2), (2, 2, 1, 1)),
    )
    cosets = [[g for g in G.elements if g[2] == t]
              for t in range(4)]
    translations = [g for g in cosets[0] if g != e]
    cross = cosets[1] + cosets[3]
    cap = min(W_MAX, 9, 12)

    legal_pairs = {
        (a, b) for a, sa in enumerate(aprofiles)
        for b, sb in enumerate(bprofiles)
        if _no_dead_lines(sa, sb) and _grid_weights(sa, sb) <= cap}
    if not legal_pairs:
        raise ValueError("no legal saturated support pairs")

    def draw(profile, policy, local, trial):
        grid = [[[] for _ in row] for row in profile]
        for i, row in enumerate(profile):
            for j, terms in enumerate(row):
                if not terms:
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    if policy in (0, 1, 3):
                        pool = cross
                    else:
                        pool = (cross if (i + j + trial) % 3 else
                                translations + cosets[2])
                    relative = local.choice(pool)
                    cell = [anchor, G.mul(anchor, relative)]
                elif policy == 1:
                    # A within-cross-coset pair changes relative
                    # translation geometry, not merely its anchor.
                    t = local.choice((1, 3))
                    g, h = local.sample(cosets[t], 2)
                    cell = [anchor, G.mul(anchor, g), G.mul(anchor, h)]
                elif policy == 3:
                    relatives = local.sample(G.elements, 3)
                    cell = [G.mul(anchor, g) for g in relatives]
                else:
                    step = local.choice(translations)
                    pool = (cross if policy == 0 else
                            cross + cosets[2])
                    outside = local.choice(pool)
                    if (trial + i + j) % 2:
                        second = G.mul(anchor, outside)
                        cell = [anchor, second, G.mul(second, step)]
                    else:
                        cell = [anchor, G.mul(anchor, step),
                                G.mul(anchor, outside)]
                assert len(set(cell)) == terms
                grid[i][j] = cell
        return grid

    def bank(side, profiles):
        entries, graph_keys = [], set()
        for profile_id, profile in enumerate(profiles):
            if not any((profile_id == a if side == "A"
                        else profile_id == b)
                       for a, b in legal_pairs):
                continue
            for policy in range(4):
                card = 4 * profile_id + policy
                local = random.Random(
                    BASE_SEED ^ (0xF90436 + 7919 * card
                                 + 104729 * int(side == "B")))
                raw, fillings = [], set()
                for trial in range(112):
                    grid = draw(profile, policy, local, trial)
                    filling = tuple(
                        tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
                    if filling in fillings:
                        continue
                    fillings.add(filling)
                    rows, width = _r15_component(G, grid, side)
                    rank = gf2_rank(rows)
                    if rank < len(grid) * q - 8:
                        continue
                    pairs = _short_pairs(_syndrome_columns(rows, width))
                    if pairs is None:
                        continue
                    del pairs
                    invariant = _r15_invariant_word(G, rows, width)
                    iw = invariant.bit_count() if invariant else width + 1
                    cycles = _r15_cycle_cost(rows)
                    cheap = (int(iw < 15),
                             (8 + 4 * policy) * cycles,
                             -min(iw, 36), trial)
                    raw.append((cheap, grid, rows, width, invariant,
                                cycles, rank))
                raw.sort(key=lambda r: r[0])

                refined = []
                for cheap, grid, rows, width, seed, cycles, rank in raw[:10]:
                    word = _r15_probe(rows, width, local, 4, seed)
                    if word:
                        refined.append(dict(
                            grid=grid, rows=rows, width=width, word=word,
                            cycles=cycles, rank=rank, profile=profile_id,
                            policy=policy, card=card, trial=cheap[-1]))
                refined.sort(key=lambda r: (
                    -r["word"].bit_count(), r["cycles"], r["trial"]))
                finalists = refined[:4]
                # Every retained finalist gets the same additional budget.
                for record in finalists:
                    record["word"] = _r15_probe(
                        record["rows"], record["width"], local, 20,
                        record["word"])
                finalists.sort(key=lambda r: (
                    -r["word"].bit_count(), r["cycles"], r["trial"]))

                retained = 0
                for record in finalists:
                    signature = _r15_key(record["rows"], record["width"])
                    if signature in graph_keys:
                        continue
                    graph_keys.add(signature)
                    entries.append(record)
                    retained += 1
                    if retained == 2:
                        break
        if len(entries) < 5:
            raise ValueError("too few independently screened constituents")
        return entries

    aa, bb = bank("A", aprofiles), bank("B", bprofiles)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            if (a["profile"], b["profile"]) not in legal_pairs:
                continue
            weak = min(a["word"].bit_count(), b["word"].bit_count())
            priority = (
                min(weak, 24),
                min(a["word"].bit_count() + b["word"].bit_count(), 48),
                -a["cycles"] - b["cycles"],
                int(a["policy"] != b["policy"]))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def reduce_word(word, stabilizers):
        for _ in range(2):
            starting = word
            ranked = sorted(stabilizers,
                            key=lambda row: (starting ^ row).bit_count())
            for row in ranked:
                trial = starting ^ row
                if trial and trial.bit_count() < word.bit_count():
                    word = trial
            short = ranked[:24]
            for i, first in enumerate(short):
                for second in short[:i]:
                    trial = starting ^ first ^ second
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def logical_reading(a, b, hx, hz, n, local):
        """Explicit original-coordinate nontrivial logical upper bounds."""
        readings = []
        ca, cb = A_SHAPE[1], B_SHAPE[1]
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _r15_span(stabilizers)
            seeds = []
            for block in range(ca if kind == "X" else cb):
                if kind == "X":
                    word = b["word"] << (block * cb * q)
                else:
                    word, source = 0, a["word"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * cb + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if _r15_outside(word, own):
                    seeds.append(word)
            if not seeds:
                return None
            seed = reduce_word(min(seeds, key=int.bit_count), stabilizers)
            word = _r15_probe(checks, n, local, 2, seed, own)
            word = reduce_word(word, stabilizers)
            assert word and _r15_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            readings.append(word.bit_count())
        return min(readings)

    results, graph_keys = [], set()
    screened = 0
    wanted = max(count + 20, 2 * count)
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert max(wx, wz) <= cap and _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Strictly above the 0.125 boundary avoids an ambiguous tie.
        if not (n == 576 and 125*n < 1000*k < 175*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _r15_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        screened += 1

        if (_short_pairs(_syndrome_columns(hx, n)) is None
                or _short_pairs(_syndrome_columns(hz, n)) is None):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quarter-turn lift failed CSS")
        local = random.Random(
            BASE_SEED ^ (0x150436 + 104729 * ai + 7919 * bi))
        reading = logical_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        # Cap unresolved large readings, not the returned actual witness.
        quality = (k * min(reading, 24) ** 2 / n,
                   min(a["word"].bit_count(), b["word"].bit_count()),
                   -a["cycles"] - b["cycles"], k)
        results.append((quality, freeze(A), freeze(B), k, reading,
                        (a["card"], b["card"]),
                        (a["word"].bit_count(), b["word"].bit_count())))
        if len(results) >= wanted:
            break
        if screened >= max(256, 8 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure publication of one immutable, full-invariant-separated design."""
    if not 0 <= idx < N_SLOTS:
        raise ValueError("slot index out of range")
    portfolio = _r15_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct target-rate quarter-turn portfolio exhausted")
    quality, A, B, k, reading, cards, component_readings = portfolio[idx]
    return dict(A=A, B=B, screen_k=k, quality=quality,
                logical_witness_upper_bound=reading,
                constituent_witness_upper_bounds=component_readings,
                design_cards=cards, complete_kernel_floor=5)


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without mutating the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity objectives on the fixed Z8 (3,8)x(3,8) triangle profile suffice to exceed distance five: all five measured codes had sector-1 logicals at d30s=5.
# REFUTED: The fixed cyclic chain/crossed gap-card catalogue supplies twenty evaluator-distinct designs: sixty evaluated slots retained seventeen classes, and all five measured readings remained six.
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
