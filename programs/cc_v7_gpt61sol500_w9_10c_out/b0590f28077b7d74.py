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
CONCEPT = ("Generalized quaternion Q32, order 32, A=(2,3), B=(2,4): independently optimized cubic-column constituents and full-CSS witness refinement should supply five measured distance readings >=14 in cell 0.150; this remains a falsifiable measurement hypothesis.")
GROUP = ("Keep the Q32 presentation and order byte-for-byte. RUN STATE reports fitness 21.4 on this destination bed and 22.0 on the related (2,3)x(3,4) bed. The current high-rate parent measured distance three on all five finalists and retained only twelve distinct codes, motivating a shape move rather than another labeling-only attempt.")
SHAPES = ("A has 96 variables/64 checks and B has 128 variables/64 checks. Factor 16 gives n=512, kb=64 and guaranteed rate 0.125; the window permits q<=37 and the frozen order 32 is legal. Verified constituent ranks 63 and 64 target full k=66, rate 0.12890625 in cell 0.150. A distance-fourteen reading then scores 25.265625 against the frontier incumbent 24.3. Parent weight-three minima occupied sector-1 Acol/Bcol blocks, especially its monomial columns; the child removes that geometry.")
FILLING = ("Twenty structural cards combine two B supports, five trinomial relative-order spectra and two B binomial-order policies. Every constituent column has three terms: A row totals four/five and B row totals six give HX<=8 and HZ<=9, below both nine and twelve. Verified constituent matrices receive exact kernel-four exclusion, explicit six-word collision tests and equal-budget information-set refinement. Complete codes enforce actual k=66, CSS, kernel-four exclusion, intrinsic uncoloured Tanner separation, and embedded plus mixed-sector logical-witness refinement with stabilizer descent. Larger readings and unseen archive novelty remain evaluator-tested, not certified.")
FAMILY_NAME = "r015_quaternion32_23x24_cubic_rankoneA_witness_portfolio_v1"

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


def _q32_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _q32_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _q32_clean4(rows, width):
    """Exact exclusion of every nonzero kernel word of weight <=4."""
    cols = _check_columns(rows, width)
    if 0 in cols or len(set(cols)) != width:
        return None
    seen = set(cols)
    seen.add(0)
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in seen:
                return None
            seen.add(syndrome)
    return cols


def _q32_six_word(cols):
    """Find an actual <=6 dependency after the exact-four screen.

    Published constituent columns all have odd degree, so odd-weight
    dependencies are impossible. Distinct equal-syndrome triples detect
    every remaining six-word. Zero means only that <=6 was excluded.
    """
    triples = {}
    bits = [1 << j for j in range(len(cols))]
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                word = bits[i] | bits[j] | bits[h]
                old = triples.get(syndrome)
                if old is not None:
                    return old ^ word
                triples[syndrome] = word
    return 0


def _q32_probe(rows, width, local, passes, seed=0, own=None):
    """Actual kernel/logical witnesses, not a lower distance bound."""
    cols = _check_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _q32_outside(word, own):
            return
        best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
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

        # Elementary exchanges retain a basis of the same kernel.
        local.shuffle(kernel)
        for i in range(len(kernel)):
            current = kernel[i]
            current_weight = current.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = current ^ other
                w = trial.bit_count()
                if 0 < w < current_weight:
                    current, current_weight = trial, w
            kernel[i] = current
            inspect(current)

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])

        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
        if weight <= 6:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _q32_outside(best, own)
    return best


def _q32_component(G, grid, side):
    """Expose only the scaffold's actual left/right constituent action."""
    zero = [[[]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _q32_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _q32_signature(rows, width):
    """Intrinsic uncoloured Tanner invariant, not a complete iso test.

    Exact check overlaps, degree-conditioned common-neighbour profiles
    and qubit degrees survive arbitrary check/qubit permutations.
    Unequal signatures separate graphs; equality is discarded
    conservatively. No CSS colouring or element indices enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(rows)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            c = (row & rows[j]).bit_count()
            overlaps.append((i, j, c))
            if c:
                relation = one if c == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, c in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(weights[j], c) + common] += 1
        profiles[j][(weights[i], c) + common] += 1

    cols = _check_columns(rows, width)
    description = (
        width, m,
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(
            (weights[i], tuple(sorted(profile.items())))
            for i, profile in enumerate(profiles))),
    )
    return sha256(repr(description).encode("ascii")).digest()


def _q32_constituents(G, side, orders, triangles):
    """Optimize twenty support/order cards, rather than publish draws."""
    profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
    )
    rotations16 = [g for g in G.elements
                   if g[1] == 0 and orders[g] == 16]
    rotations8 = [g for g in G.elements
                  if g[1] == 0 and orders[g] == 8]
    outside4 = [g for g in G.elements if g[1] == 1]
    bank, signatures = [], set()

    for card in range(20):
        profile = card // 10
        policy = (card % 10) // 2
        ratio_policy = card % 2
        support = ((0, 2, 2), (3, 1, 1)) if side == "A" \
            else profiles[profile]
        local = random.Random(
            BASE_SEED ^ (0x321524 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        raw, fillings = [], set()

        def draw_cell(count, i, j):
            if not count:
                return []
            anchor = G.pick(local)
            if count == 1:
                return [anchor]
            if count == 2:
                if side == "A":
                    # Together these directions generate Q32, while the
                    # bottom trinomial is a unit in the 2-group algebra.
                    pool = rotations16 if j == 1 else outside4
                else:
                    pool = rotations16 if ratio_policy == 0 else rotations8
                relative = local.choice(pool)
                return [anchor, G.mul(anchor, relative)]
            kind = policy if side == "A" else (policy + i + j) % 5
            r, s = local.choice(triangles[kind])
            return [anchor, G.mul(anchor, r), G.mul(anchor, s)]

        for trial in range(80):
            grid = [
                [draw_cell(count, i, j) for j, count in enumerate(row)]
                for i, row in enumerate(support)
            ]
            fingerprint = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if fingerprint in fillings:
                continue
            fillings.add(fingerprint)
            if any(len(set(cell)) != len(cell)
                   for row in grid for cell in row):
                raise ValueError("duplicate element in cubic card")
            rows, width = _q32_component(G, grid, side)
            expected = 63 if side == "A" else 64
            if gf2_rank(rows) != expected:
                continue
            cols = _q32_clean4(rows, width)
            if cols is None:
                continue
            if any(col.bit_count() != 3 for col in cols):
                raise ValueError("constituent column degree changed")
            raw.append((_q32_cycles(rows), trial, grid, rows, cols))

        raw.sort(key=lambda record: record[:2])
        shortlist = []
        for cost, trial, grid, rows, cols in raw[:12]:
            width = len(cols)
            witness = _q32_probe(rows, width, local, 3)
            if witness:
                shortlist.append(dict(
                    grid=grid, rows=rows, cols=cols,
                    witness=witness, cost=cost, trial=trial, card=card))
        shortlist.sort(key=lambda record: (
            -record["witness"].bit_count(), record["cost"], record["trial"]))

        finalists = shortlist[:4]
        for record in finalists:
            six = _q32_six_word(record["cols"])
            seed = record["witness"]
            if six and six.bit_count() < seed.bit_count():
                seed = six
            record["floor"] = 6 if six else 8
            record["witness"] = _q32_probe(
                record["rows"], len(record["cols"]), local, 12, seed)
            record["reading"] = record["witness"].bit_count()
        finalists.sort(key=lambda record: (
            -record["reading"], -record["floor"],
            record["cost"], record["trial"]))

        kept = 0
        for record in finalists:
            key = _q32_signature(record["rows"], len(record["cols"]))
            if key in signatures:
                continue
            signatures.add(key)
            bank.append(record)
            kept += 1
            if kept == 2:
                break
    return bank


def _q32_descend(word, stabilizers):
    """Adding stabilizers preserves the logical class; search is heuristic."""
    for _ in range(3):
        starting = word
        weight = word.bit_count()
        for row in stabilizers:
            trial = starting ^ row
            if trial.bit_count() < weight:
                word, weight = trial, trial.bit_count()
        moves = [row for row in stabilizers if row & starting]
        for i, first in enumerate(moves):
            for second in moves[:i]:
                trial = starting ^ first ^ second
                if trial.bit_count() < weight:
                    word, weight = trial, trial.bit_count()
        if word == starting:
            break
    return word


def _q32_embedded(record, kind, q, checks, stabilizers, own):
    """Verify constituent embeddings in the evaluator's column order."""
    best = 0
    count = A_SHAPE[1] if kind == "X" else B_SHAPE[1]
    for block in range(count):
        if kind == "X":
            word = record["witness"] << (block * B_SHAPE[1] * q)
        else:
            word, source = 0, record["witness"]
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length() - 1, q)
                word |= 1 << ((j * B_SHAPE[1] + block) * q + h)
                source ^= bit
        if any((row & word).bit_count() % 2 for row in checks):
            raise ValueError("verified constituent embedding failed")
        if not _q32_outside(word, own):
            continue
        word = _q32_descend(word, stabilizers)
        assert word and _q32_outside(word, own)
        if not best or word.bit_count() < best.bit_count():
            best = word
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _q32_cubic_portfolio(G):
    """Immutable deterministic portfolio; publication order changes nothing."""
    from itertools import combinations

    if G.q != 32:
        raise ValueError("frozen cubic portfolio requires Q32")

    def order(g):
        power = G.e
        for length in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return length
        raise ValueError("invalid finite-group order")

    orders = {g: order(g) for g in G.elements}
    spectra = ((16, 16, 8), (16, 16, 4),
               (4, 4, 16), (4, 4, 8), (4, 4, 4))
    triangles = [[] for _ in spectra]
    nonidentity = [g for g in G.elements if g != G.e]
    for r, s in combinations(nonidentity, 2):
        spectrum = tuple(sorted((
            orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
        for policy, requested in enumerate(spectra):
            if spectrum == tuple(sorted(requested)):
                triangles[policy].append((r, s))
    if any(not pool for pool in triangles):
        raise ValueError("required Q32 triangle spectrum absent")

    aa = _q32_constituents(G, "A", orders, triangles)
    bb = _q32_constituents(G, "B", orders, triangles)
    if not aa or not bb:
        raise ValueError("no rank-controlled cubic constituent bank")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                min(a["floor"], b["floor"]),
                -a["cost"] - b["cost"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda plan: (plan[0], -plan[1], -plan[2]),
               reverse=True)

    results, signatures = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb):
            continue
        if _grid_weights(sa, sb) > min(9, W_MAX):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 512 or k != 66:
            continue
        if not (125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
            continue

        signature = _q32_signature(hx + hz, n)
        if signature in signatures:
            continue
        signatures.add(signature)
        tested += 1
        if _q32_clean4(hx, n) is None or _q32_clean4(hz, n) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified cubic lift failed CSS")

        own_x, own_z = _q32_basis(hx), _q32_basis(hz)
        xseed = _q32_embedded(b, "X", G.q, hz, hx, own_x)
        zseed = _q32_embedded(a, "Z", G.q, hx, hz, own_z)
        if not xseed or not zseed:
            continue
        local = random.Random(
            BASE_SEED ^ (0x32C151 + 65537 * ai + 1009 * bi))
        xword = _q32_probe(hz, n, local, 2, xseed, own_x)
        zword = _q32_probe(hx, n, local, 2, zseed, own_z)
        xword = _q32_descend(xword, hx)
        zword = _q32_descend(zword, hz)
        assert _q32_outside(xword, own_x)
        assert _q32_outside(zword, own_z)
        assert all((row & xword).bit_count() % 2 == 0 for row in hz)
        assert all((row & zword).bit_count() % 2 == 0 for row in hx)
        reading = min(xword.bit_count(), zword.bit_count())

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        results.append((
            (k * reading * reading / n, reading, preliminary),
            freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (a["floor"], b["floor"])))
        if len(results) >= N_SLOTS:
            break
        if tested >= max(240, 6 * N_SLOTS):
            break

    results.sort(key=lambda record: record[0], reverse=True)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure lookup of an optimized, invariant-separated full filling."""
    portfolio = _q32_cubic_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("rank-controlled Q32 cubic portfolio exhausted")
    quality, A, B, k, reading, cards, floors = portfolio[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "logical_witness_upper_bound": reading,
        "structural_cards": cards,
        "constituent_kernel_floors": floors,
        "quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without changing shared state."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 unit/binomial catalogue's intersection and qubit-degree gates supply twenty evaluator-distinct codes: the measured sixty slots retained twelve classes and forty-eight iso_dup slots, with all five measured distances remaining three.
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
