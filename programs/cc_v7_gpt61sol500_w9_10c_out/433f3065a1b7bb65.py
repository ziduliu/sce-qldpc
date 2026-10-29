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
CONCEPT = ("Generalized quaternion Q32, order 32, A=(2,3), B=(2,4): independently optimized saturated constituents followed by complete-matrix logical screening should supply five novel measured codes averaging a score above 25.2 in cell 0.150; this is a measurement hypothesis, not a distance certificate.")
GROUP = ("Keep the Q32 presentation and order unchanged. The new factor sixteen gives legal n=512. This exact group/shape bed measured fitness 21.4, versus the 0.150 frontier incumbent's 25.2. The parent's one-row pivot construction certifies distance three and already gives five full scoring contributions, so merely improving its catalogue yield cannot increase that score.")
SHAPES = ("Use unequal (2,3)x(2,4) constituents, with 96 variables/64 checks and 128 variables/64 checks. n=512, kb=64 and guaranteed rate 1/8. Verified ranks require 66<=k and 1000*k<175*n, placing every published code in cell 0.150. The parent reported weight-three minima entirely in sector-1 Acol/Bcol blocks. Two-row mixed constituents replace its explicit pivot/binomial cancellation geometry.")
FILLING = ("Every constituent row has five terms and every column at most four, so BOTH CSS check maxima remain nine. Structural cards vary three A profiles, four B profiles and four quaternion coset prescriptions. Optimize A and B independently using verified lifts, exact constituent kernel rejection through four and equal-budget kernel-witness searches. Complete pairs pass verified ranks, CSS, exact logical rejection through four and intrinsic uncoloured Tanner invariants. Explicit embedded witnesses receive stabilizer reduction and randomized mixed-sector probing before portfolio ranking. Larger readings remain upper bounds; archive novelty and improvement are evaluator-tested hypotheses.")
FAMILY_NAME = "r015_quaternion32_23x24_saturated_independent_logicalportfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if q != 32:
        raise ValueError("Q32 has order 32")
    # x^16=1, y^2=x^8, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        a, b = g
        return ((a + 8 * (b // 2)) % 16, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 8 * b * d) % 16,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 16, 0) if b == 0 else ((a + 8) % 16, 1)

    def idx(g):
        a, b = red(g)
        return 16 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _generator_cards(G):
    """Enumerate inverse-normalized generating triples, not label draws."""
    from itertools import combinations

    # One representative of each inverse pair, excluding e and x^8.
    representatives = [(a, 0) for a in range(1, 8)]
    representatives += [(a, 1) for a in range(8)]

    def order(g):
        h, count = G.e, 0
        while True:
            h = G.mul(h, g)
            count += 1
            if h == G.e:
                return count

    cards = []
    for triple in combinations(representatives, 3):
        reached, frontier = {G.e}, [G.e]
        while frontier:
            h = frontier.pop()
            for g in triple:
                t = G.mul(h, g)
                if t not in reached:
                    reached.add(t)
                    frontier.append(t)
        if len(reached) == G.q:
            cards.append(triple)
    cards.sort(key=lambda t: (tuple(sorted(order(g) for g in t)),
                              sum(g[1] for g in t), t))
    return cards


def _unit_binomial_cells(G, triple):
    return [[[G.e]] + [[G.e, g] for g in triple]]


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


def _check_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _no_weight_two_kernel(rows, n):
    """Exact exclusion of every nonzero kernel vector of weight <=2."""
    columns = _check_columns(rows, n)
    seen = set(columns)
    return 0 not in seen and len(seen) == n


def _degree_signature(hx, hz, n):
    """Additional intrinsic Tanner invariants; no X/Z colouring assumed.

    Project only degree-three qubits onto checks. Quibit degree is
    invariant under any Tanner relabeling, so different signatures
    certify non-isomorphic full graphs.
    """
    rows = hx + hz
    columns = _check_columns(rows, n)
    degree_counts = {}
    adj = [0] * len(rows)
    for c in columns:
        degree = c.bit_count()
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        if degree == 3:
            bits = c
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= c ^ bit
                bits ^= bit

    profiles = []
    for i, neighbors in enumerate(adj):
        histogram = {}
        for j, other in enumerate(adj):
            if i == j:
                continue
            key = (((neighbors >> j) & 1),
                   other.bit_count(),
                   (neighbors & other).bit_count(),
                   rows[j].bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        profiles.append((rows[i].bit_count(), neighbors.bit_count(),
                         tuple(sorted(histogram.items()))))
    return (tuple(sorted(degree_counts.items())), tuple(sorted(profiles)))


def _verify_three_witness(G, B, hx, hz):
    """Verify a sector-1 X logical of weight three in the actual lift."""
    q = G.q
    g = B[0][1][1]
    # B=[1, 1+g, ...]. Its right-lifted binomial column at h=e
    # is corrected by the two unit-column qubits at e and g^-1.
    word = ((1 << (q + G.idx(G.e)))
            ^ (1 << G.idx(G.e))
            ^ (1 << G.idx(G.inv(g))))
    if word.bit_count() != 3:
        return False
    if any((word & row).bit_count() % 2 for row in hz):
        return False

    pivots = {}
    for row in hx:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _sat_span(rows):
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


def _sat_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _sat_kernel4(rows, width):
    """Exact constituent kernel exclusion through weight four."""
    cols = _check_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return None
        seen.add(syndrome)
    squares = 0
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in seen:
                return None
            seen.add(syndrome)
            overlap = (first & second).bit_count()
            squares += overlap * (overlap - 1) // 2
    return cols, squares


def _sat_probe(local, cols, passes, seed=0):
    """Heuristic kernel upper bound carrying an actual witness."""
    width = len(cols)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
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

        local.shuffle(nulls)
        for i in range(len(nulls)):
            word = nulls[i]
            w = word.bit_count()
            for j, other in enumerate(nulls):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            nulls[i] = word
            inspect(word)

        for i, first in enumerate(nulls):
            for second in nulls[:i]:
                inspect(first ^ second)
        short = sorted(nulls, key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])

        for _ in range(2):
            starting = best
            for i, first in enumerate(nulls):
                inspect(starting ^ first)
                for second in nulls[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
        if weight <= 8:
            break

    if not best:
        raise ValueError("constituent kernel witness absent")
    syndrome = 0
    source = best
    while source:
        bit = source & -source
        syndrome ^= cols[bit.bit_length() - 1]
        source ^= bit
    assert syndrome == 0
    return weight, best


def _sat_no_logical4(checks, stabilizers, width):
    """Exact logical exclusion through four, including mixed sectors."""
    cols = _check_columns(checks, width)
    basis = _sat_span(stabilizers)
    seen = {0: 0}

    def accept(syndrome, word):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
            return True
        return not _sat_outside(word ^ old, basis)

    for j, syndrome in enumerate(cols):
        if not accept(syndrome, 1 << j):
            return False
    for j, syndrome in enumerate(cols):
        bit = 1 << j
        for i in range(j):
            if not accept(syndrome ^ cols[i], bit | (1 << i)):
                return False
    return True


def _sat_graph_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Unequal keys certify non-isomorphism. Equal keys are rejected
    conservatively; equality is not an isomorphism certificate.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _check_columns(rows, width)
    endpoints = []
    shared = Counter()
    for col in cols:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        endpoints.append(checks)
        for i, a in enumerate(checks):
            for b in checks[:i]:
                shared[(min(a, b), max(a, b))] += 1

    adjacency = [{} for _ in rows]
    for (a, b), multiplicity in shared.items():
        adjacency[a][b] = multiplicity
        adjacency[b][a] = multiplicity
    masks = [sum(1 << j for j in neighbors)
             for neighbors in adjacency]
    features = []
    for i, neighbors in enumerate(adjacency):
        profile = Counter(
            (rows[j].bit_count(), multiplicity,
             (masks[i] & masks[j]).bit_count())
            for j, multiplicity in neighbors.items())
        features.append((
            rows[i].bit_count(), masks[i].bit_count(),
            tuple(sorted(profile.items()))))

    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(4):
        qfeatures = [tuple(sorted(colors[r] for r in checks))
                     for checks in endpoints]
        qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        neighborhoods = [[] for _ in rows]
        for checks, feature in zip(endpoints, qfeatures):
            for r in checks:
                neighborhoods[r].append(qp[feature])
        refined = [(colors[r], tuple(sorted(neighborhoods[r])))
                   for r in range(len(rows))]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]

    invariant = (
        width, len(rows),
        tuple(sorted(Counter(len(c) for c in endpoints).items())),
        tuple(history))
    return sha256(repr(invariant).encode("ascii")).digest()


def _sat_logical_reading(G, a, b, hx, hz, local):
    """Search actual nontrivial logicals with the same budget per pair.

    Embedded constituent witnesses, randomized mixed-sector null words
    and stabilizer moves all use the evaluator's exact coordinates.
    """
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    width = (ca * cb + ra * rb) * q
    overall = width + 1

    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _sat_span(stabilizers)
        best, weight = 0, width + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if word and w < weight and _sat_outside(word, own):
                best, weight = word, w

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
            inspect(word)

        cols = _check_columns(checks, width)
        order = list(range(width))
        for _ in range(2):
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

            short = sorted(nulls, key=int.bit_count)[:24]
            reduced = []
            for word in short:
                w = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
                inspect(word)
                if _sat_outside(word, own):
                    reduced.append(word)
            for i, first in enumerate(reduced):
                for second in reduced[:i]:
                    inspect(first ^ second)
            tiny = reduced[:16]
            for i, first in enumerate(tiny):
                for j in range(i):
                    pair = first ^ tiny[j]
                    for h in range(j):
                        inspect(pair ^ tiny[h])

            if best:
                starting = best
                for i, first in enumerate(stabilizers):
                    inspect(starting ^ first)
                    for second in stabilizers[:i]:
                        inspect(starting ^ first ^ second)

        if not best:
            return None
        assert _sat_outside(best, own)
        assert all((row & best).bit_count() % 2 == 0
                   for row in checks)
        overall = min(overall, weight)
    return overall


from functools import lru_cache


@lru_cache(maxsize=1)
def _sat_portfolio(G, count):
    """Deterministic immutable portfolio; memoization only saves work."""
    if G.q != 32:
        raise ValueError("the frozen Q32 group requires order thirty-two")
    cap = min(9, W_MAX)
    aprofiles = (
        ((1, 2, 2), (2, 1, 2)),
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
    )
    bprofiles = (
        ((2, 1, 1, 1), (2, 1, 1, 1)),
        ((2, 1, 1, 1), (1, 2, 1, 1)),
        ((3, 1, 1, 0), (1, 1, 1, 2)),
        ((3, 1, 1, 0), (0, 2, 1, 2)),
    )
    for sa in aprofiles:
        for sb in bprofiles:
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) <= cap

    rotations = [(a, 0) for a in range(1, 16) if a != 8]
    crosses = [(a, 1) for a in range(16)]
    odd_rotations = [(a, 0) for a in range(1, 16, 2)]

    def draw(local, profile, policy, trial):
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, terms in enumerate(counts):
                if not terms:
                    row.append([])
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    if policy == 0:
                        pool = odd_rotations
                    elif policy == 1:
                        pool = crosses
                    elif policy == 2:
                        pool = crosses if (i + j + trial) % 2 else rotations
                    else:
                        pool = rotations + crosses
                    relative = local.choice(pool)
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    rotation = local.choice(
                        odd_rotations if policy < 2 else rotations)
                    cross = local.choice(crosses)
                    if (i + j + trial + policy) % 2:
                        second = G.mul(anchor, cross)
                        cell = [anchor, second, G.mul(second, rotation)]
                    else:
                        cell = [anchor, G.mul(anchor, rotation),
                                G.mul(anchor, cross)]
                assert len({G._key(g) for g in cell}) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def component(grid, side):
        """Extract the exact constituent with a zero auxiliary cell."""
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
            width = A_SHAPE[1] * G.q
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
            width = B_SHAPE[1] * G.q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def bank(side, profiles):
        entries, fillings = [], set()
        for pi, profile in enumerate(profiles):
            for policy in range(4):
                local = random.Random(
                    BASE_SEED ^ (0x532340 + 1009 * pi + 65537 * policy
                                 + (104729 if side == "B" else 0)))
                raw = []
                for trial in range(96):
                    grid = draw(local, profile, policy, trial)
                    filling = tuple(
                        tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
                    if filling in fillings:
                        continue
                    fillings.add(filling)
                    rows, width = component(grid, side)
                    stats = _sat_kernel4(rows, width)
                    if stats is None:
                        continue
                    cols, squares = stats
                    raw.append((squares, trial, grid, rows, cols))
                raw.sort(key=lambda item: item[:2])

                shortlist = []
                for squares, trial, grid, rows, cols in raw[:8]:
                    reading, word = _sat_probe(local, cols, 4)
                    shortlist.append(dict(
                        grid=grid, rows=rows, cols=cols, squares=squares,
                        trial=trial, card=(pi, policy),
                        reading=reading, word=word))
                shortlist.sort(key=lambda r: (
                    -r["reading"], r["squares"], r["trial"]))

                # Every retained finalist receives the same 4+20 passes.
                for record in shortlist[:2]:
                    reading, word = _sat_probe(
                        local, record["cols"], 20, record["word"])
                    record["reading"], record["word"] = reading, word
                    entries.append(record)
        entries.sort(key=lambda r: (
            -r["reading"], r["squares"], r["card"], r["trial"]))
        if not entries:
            raise ValueError("no constituent passed the exact kernel-four gate")
        return entries

    aa, bb = bank("A", aprofiles), bank("B", bprofiles)
    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"],
         a["squares"] + b["squares"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    results, graph_keys = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 512 or not (66 <= k and 125*n <= 1000*k < 175*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _sat_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        if not (_sat_no_logical4(hz, hx, n)
                and _sat_no_logical4(hx, hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified saturated lift failed CSS")
        local = random.Random(BASE_SEED ^ (0x53234D + 1009*ai + bi))
        reading = _sat_logical_reading(G, a, b, hx, hz, local)
        if reading is None:
            continue
        quality = (k * reading * reading / n, reading, k,
                   -a["squares"] - b["squares"])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]), signature.hex()))
        if len(results) >= max(count, 5) + 12:
            break

    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable optimized portfolio."""
    entries = _sat_portfolio(G, N_SLOTS)
    if idx >= len(entries):
        raise ValueError("distinct screened saturated portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k, "quality": quality,
        "structural_cards": cards,
        "component_witness_readings": components,
        "logical_witness_reading": reading,
        "logical_weights_through_four_excluded": True,
        "intrinsic_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without changing shared state."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: Q32 generating-triple selection with intersection and qubit-degree gates supplies twenty evaluator-distinct slots: the reported sixty-slot evaluation retained twelve distinct codes, with all five measured distances remaining three.
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
