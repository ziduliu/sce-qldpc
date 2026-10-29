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
CONCEPT = ("C3 semidirect C2 with inversion action 2, order 6, A=(4,10), B=(3,8): four-row constituent geometry and exact rank-deficit selection should retain distance-six readings and supply five finalists with k>=190, exceeding the 0.350 frontier; this is a falsifiable measurement hypothesis.")
GROUP = ("Keep the metacyclic semidirect-product construction rule, resizing only its parameters to m=3,r=2,s=2. Factor 92 makes the parent's order 12 invalid; order six gives n=552 and 2^2=1 modulo 3. This precise bed is absent from BEDS. The parent measured five [552,180,6] codes, while the cell frontier scored 12.329. Neither noncommutation nor a rank deficit guarantees transferred distance.")
SHAPES = ("A=(4,10), B=(3,8) give 60-variable/24-check and 48-variable/18-check constituents. Factor 92 permits q<=6; q=6 gives n=552, kb=180 and guaranteed rate 15/46=0.326087. The complete check counts remain 192 and 180, but A now has four protograph rows. Parent minima broadly occupied sector-1 Acol/Bcol blocks, with no measured sector-2 minimum. Verified ranks retain [0.325,0.375); a surviving distance-six reading at k=190 would score 12.3913.")
FILLING = ("Twenty cards per side vary intrinsic supports and relative-order policies. A starts from six tetrahedral matching columns and four degree-three faces; incidence exchanges produce additional supports. B uses one singleton, four matching columns and three cubic columns, placing the singleton away from the parallel bundle. Every component row totals six and every column has at most three terms, so BOTH CSS maxima are nine. Exact constituent spectra exclude weights through five when possible, and selection favors useful rank deficits. Verified full lifts enforce target-band ranks, CSS, kernel-four rejection and mixed-sector weight-five screens. Immutable publication and unequal uncoloured Tanner invariants separate slots. Kernel-four-clean fallbacks are retained; witness readings and archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r035_metacyclic3by2_s2_q6_410x38_rank_surplus_portfolio_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 6

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the resized metacyclic group requires order 6")
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 92 excludes order 12; s^r=2^2=1 mod 3.
    m, r, s = 3, 2, 2
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

A_SHAPE = (4, 10)              # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


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


def _inactive_order55_catalogue(rng, G, idx):
    """Historical reference only; the order-111 NEW_CTX never calls this."""
    from itertools import combinations, product

    def clean4(rows, width):
        """Exactly exclude all nonzero kernel supports of weights 1..4."""
        cols = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= tag
                row ^= bit
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return False
        pairs = set()
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        # Distinct nonzero columns make a pair-pair collision a
        # four-distinct-qubit support. Pair-single collisions detect three.
        return True

    # A binomial is an undirected Cayley edge. Enumerate ALL triples,
    # rather than sampling a small bank and correlating its two sides.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if G._key(g) != G._key(G.e)
    })
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    automorphisms = []
    for u in range(1, 11):
        for v in range(11):
            image = {}
            for label in reps:
                a, b = G.elements[label]
                g = ((u * a + v * sums[b]) % 11, b)
                image[label] = min(G.idx(g), G.idx(G.inv(g)))
            automorphisms.append(image)

    keys = set()
    for triple in combinations(reps, 3):
        generators = [G.elements[t] for t in triple]
        # Commuting distinct edge generators immediately make a square.
        if any(G.mul(generators[i], generators[j])
               == G.mul(generators[j], generators[i])
               for i in range(3) for j in range(i)):
            continue
        key = min(tuple(sorted(image[t] for t in triple))
                  for image in automorphisms)
        keys.add(key)

    bank = []
    for key in sorted(keys):
        grid = [[[G.e, G.elements[t]] for t in key]]
        rows = _component_rows(G, "A", grid)
        if gf2_rank(rows) != G.q - 1:
            continue
        if not clean4(rows, 3 * G.q):
            continue
        quotient = tuple(sorted(
            min(G.elements[t][1], (-G.elements[t][1]) % 5)
            for t in key))
        bank.append((quotient, key, grid))
    if not bank:
        raise ValueError("no connected kernel-four-clean constituent")

    graph_keys, filling_keys = set(), set()
    accepted = 0

    def consider(A, B, prescription):
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
        if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
            return None
        # Constituent cleanliness does not exclude mixed-sector words.
        # Both complete centralizers must pass, including sector 2.
        if not clean4(hx, n) or not clean4(hz, n):
            return None
        signature = _tanner_signature(hx, hz)
        if signature in graph_keys:
            return None
        graph_keys.add(signature)

        if accepted == idx:
            assert n == 550 and css_ok(hx, hz)
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "centralizer_screens": (5, 5),
                    "prescription": prescription}
        accepted += 1
        return None

    # Inner scalings are powers of 3. The multiplier 2 represents the
    # other outer-automorphism coset. Its relative orientation is tested,
    # not assumed to produce a different code.
    plans = []
    for a in range(len(bank)):
        for b in range(a, len(bank)):
            for twist in (1, 2):
                plans.append((bank[a][0] == bank[b][0], a, b, twist))
    plans.sort()
    for _, a, b, twist in plans:
        A = bank[a][2]
        B = [[[
            G.elements[G.idx(((twist * g[0]) % 11, g[1]))]
            for g in cell
        ] for cell in bank[b][2][0]]]
        result = consider(A, B, ("cross-binomial", a, b, twist))
        if result is not None:
            return result

    # Additional structural cards use one noncommuting trinomial and
    # two binomials. Total counts are (6,7), so both weight maxima are 9.
    # Cards prescribe quotient coordinates; intrinsic invariants and
    # exact screens, not random label fingerprints, select the catalogue.
    cards = sorted(
        (b0, b1, c0, c1)
        for b0, b1, c0, c1 in product(range(5), repeat=4)
        if b0 <= b1 and c0 <= c1
        and (b0 or b1) and (c0 or c1))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55C405)
    for attempt in range(4096):
        card = attempt % len(cards)
        b0, b1, c0, c1 = cards[card]
        g, h = local.choice(pools[b0]), local.choice(pools[b1])
        if g == h or G.mul(g, h) == G.mul(h, g):
            continue
        s, t = local.choice(pools[c0]), local.choice(pools[c1])
        if G.mul(s, t) == G.mul(t, s):
            continue
        B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
        if not clean4(_component_rows(G, "B", B), 3 * G.q):
            continue
        a = (attempt // len(cards) + card) % len(bank)
        result = consider(bank[a][2], B, ("trinomial-card", a, card))
        if result is not None:
            return result
    raise ValueError("strict full-kernel-four-clean catalogue exhausted")


def _inactive_order111_context(rng, G, idx):
    """Historical reference only; the resized entry point never calls this.

    Its fixed dimensions and ownership model are not used to publish
    any cells of the new shapes.
    """
    from collections import Counter
    from hashlib import sha256

    assert G.q == 111 and 0 <= idx < N_SLOTS
    q, width = G.q, 2 * G.q
    card, epoch = idx % 20, idx // 20
    local = random.Random(BASE_SEED * 10007 + idx + 0x111C3)
    counts = [[3, 3]]
    wx = sum(counts[0]) + max(counts[0])
    wz = sum(counts[0]) + max(counts[0])
    assert wx == wz == 9
    assert max(wx, wz) <= min(9, 12, W_MAX)
    assert _no_dead_lines(counts, counts)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def component(grid, side):
        unit = [[[G.e]]]
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = hx[:q]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = hz[:q]
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    def draw(side, trial):
        # Ten quotient prescriptions times two within-coset-pair policies.
        # Translations are trials; intrinsic matrices, not translations,
        # determine ownership and selection.
        prescriptions = (
            (1, 1), (1, 2), (2, 2), (2, 1), (1, 1),
            (1, 2), (2, 2), (2, 1), (1, 2), (2, 1),
        )
        bpair = prescriptions[card % 10]
        row = []
        for j in range(2):
            first = G.pick(local)
            b = bpair[(j + int(side == "B")) % 2]
            if (card % 10 >= 4 and (trial + epoch) % 3 == 0):
                b = 3 - b
            step = local.randrange(1, 37)
            cross = (local.randrange(37), b)
            second = G.mul(first, cross)
            rotation = (step, 0)
            if (card // 10 + j + int(side == "B") + epoch) % 2:
                cell = [first, second, G.mul(second, rotation)]
            else:
                cell = [first, G.mul(first, rotation), second]
            assert len(set(cell)) == 3
            row.append(cell)
        return [row]

    def cycle_cost(rows):
        cols = columns(rows, width)
        shared = Counter()
        for syndrome in cols:
            neighbours = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbours.append(bit.bit_length() - 1)
                syndrome ^= bit
            for i, a in enumerate(neighbours):
                for b in neighbours[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        return sum(c * (c - 1) // 2 for c in shared.values())

    def probe(rows, passes, seed=0):
        """Return an actual kernel witness; never a distance lower bound."""
        cols = columns(rows, width)
        best = seed
        weight = seed.bit_count() if seed else width + 1

        def inspect(word):
            nonlocal best, weight
            if word and word.bit_count() < weight:
                best, weight = word, word.bit_count()

        order = list(range(width))
        for _ in range(passes):
            local.shuffle(order)
            pivots, basis = {}, []
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
                    basis.append(word)
                    inspect(word)

            # Elementary exchanges preserve the entire kernel.
            local.shuffle(basis)
            for i in range(len(basis)):
                w = basis[i].bit_count()
                for j in range(len(basis)):
                    if i == j:
                        continue
                    trial = basis[i] ^ basis[j]
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        basis[i], w = trial, tw
                inspect(basis[i])

            for i, first in enumerate(basis):
                for second in basis[:i]:
                    inspect(first ^ second)
            short = sorted(basis, key=int.bit_count)[:24]
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
            if weight <= 8:
                break
        assert best
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return weight, best

    def pool(side):
        raw, used = [], set()
        for trial in range(64):
            grid = draw(side, trial)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for cell in grid[0])
            if key in used:
                continue
            used.add(key)
            rows = component(grid, side)
            # Odd augmentation does not force full rank; measure it.
            rank = gf2_rank(rows)
            if rank < q - 6:
                continue
            raw.append((cycle_cost(rows), trial, grid, rows))
        raw.sort(key=lambda item: item[:2])
        shortlisted = []
        for cost, trial, grid, rows in raw[:24]:
            if _kernel_profile(rows, width)[0] != 5:
                continue
            reading, witness = probe(rows, 6)
            shortlisted.append(dict(
                grid=grid, rows=rows, cost=cost, trial=trial,
                reading=reading, witness=witness))
        shortlisted.sort(key=lambda r: (
            r["reading"], -r["cost"], -r["trial"]), reverse=True)
        finalists = shortlisted[:8]
        for record in finalists:
            record["reading"], record["witness"] = probe(
                record["rows"], 18, record["witness"])
        finalists.sort(key=lambda r: (
            r["reading"], -r["cost"], -r["trial"]), reverse=True)
        return finalists

    def ownership(hx, hz, n):
        """Uncoloured Tanner invariant, including incidence refinement.

        Isomorphic Tanner graphs always receive the same owner. Equal
        invariants are inconclusive, but different owners certify that
        distinct slots cannot be isomorphic.
        """
        rows = hx + hz
        cols = columns(rows, n)
        incidence, shared = [], Counter()
        for syndrome in cols:
            neighbours = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbours.append(bit.bit_length() - 1)
                syndrome ^= bit
            incidence.append(neighbours)
            for i, a in enumerate(neighbours):
                for b in neighbours[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        adjacency = [{} for _ in rows]
        for (a, b), multiplicity in shared.items():
            adjacency[a][b] = multiplicity
            adjacency[b][a] = multiplicity
        masks = [sum(1 << j for j in neighbours)
                 for neighbours in adjacency]
        features = []
        for i, neighbours in enumerate(adjacency):
            histogram = Counter(
                (rows[j].bit_count(), multiplicity,
                 (masks[i] & masks[j]).bit_count())
                for j, multiplicity in neighbours.items())
            features.append((rows[i].bit_count(),
                             tuple(sorted(histogram.items()))))
        palette = {feature: c for c, feature in
                   enumerate(sorted(set(features)))}
        colors = [palette[feature] for feature in features]
        history = [tuple(sorted(Counter(features).items()))]
        for _ in range(3):
            qfeatures = [tuple(sorted(colors[r] for r in neighbours))
                         for neighbours in incidence]
            qpalette = {feature: c for c, feature in
                        enumerate(sorted(set(qfeatures)))}
            profiles = [[] for _ in rows]
            for neighbours, feature in zip(incidence, qfeatures):
                for r in neighbours:
                    profiles[r].append(qpalette[feature])
            refined = [(colors[r], tuple(sorted(profiles[r])))
                       for r in range(len(rows))]
            history.append((tuple(sorted(Counter(qfeatures).items())),
                            tuple(sorted(Counter(refined).items()))))
            palette = {feature: c for c, feature in
                       enumerate(sorted(set(refined)))}
            colors = [palette[feature] for feature in refined]
        digest = sha256(repr((n, len(rows), history)).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS, digest

    def quantum_reading(a, b, hx, hz):
        """Verified nontrivial witnesses, with equal stabilizer descent."""
        best, found = 556, False
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = span(stabilizers)
            for block in range(2):
                if kind == "X":
                    word = b["witness"] << (block * width)
                else:
                    word, source = 0, a["witness"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 2 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not outside(word, own):
                    continue
                found = True
                weight = word.bit_count()
                for _ in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        tw = trial.bit_count()
                        if tw < weight:
                            word, weight = trial, tw
                        for second in stabilizers[:i]:
                            trial = starting ^ first ^ second
                            tw = trial.bit_count()
                            if tw < weight:
                                word, weight = trial, tw
                    if word == starting:
                        break
                assert word and outside(word, own)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best if found else None

    winner, tested = None, set()
    for batch in range(8):
        aa, bb = pool("A"), pool("B")
        plans = [(min(a["reading"], b["reading"]),
                  a["reading"] + b["reading"],
                  -a["cost"] - b["cost"], ai, bi)
                 for ai, a in enumerate(aa)
                 for bi, b in enumerate(bb)]
        plans.sort(reverse=True)
        survivors = 0
        for _, _, _, ai, bi in plans:
            a, b = aa[ai], bb[bi]
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 555 and k > 0
                    and 175 * n <= 1000 * k < 225 * n):
                continue
            owner, signature = ownership(hx, hz, n)
            if owner != idx or signature in tested:
                continue
            tested.add(signature)
            # Complete screens include all mixed-sector words <=4.
            if (_kernel_profile(hx, n)[0] != 5
                    or _kernel_profile(hz, n)[0] != 5):
                continue
            assert css_ok(hx, hz)
            reading = quantum_reading(a, b, hx, hz)
            if reading is None:
                continue
            quality = (k * reading * reading / n, reading,
                       min(a["reading"], b["reading"]),
                       -a["cost"] - b["cost"])
            result = dict(
                A=A, B=B, screen_k=k, quality=quality,
                component_witness_readings=(a["reading"], b["reading"]),
                logical_witness_upper_bound=reading,
                intrinsic_owner=owner,
                intrinsic_signature=signature.hex(),
                design_card=(card, epoch), complete_kernel_floor=5)
            if winner is None or quality > winner["quality"]:
                winner = result
            survivors += 1
            if survivors >= 3:
                break
        if winner is not None:
            break

    if winner is None:
        raise ValueError("no independently refined cubic pair in ownership bin")
    hx, hz, n = lifted_matrices(G, winner["A"], winner["B"])
    assert css_ok(hx, hz)
    assert lifted_k(G, winner["A"], winner["B"]) == winner["screen_k"]
    assert max(row.bit_count() for row in hx + hz) == 9
    return winner


def _l2_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _l2_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _l2_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _l2_floor(rows, width, through_six=False):
    """Exact truncated kernel screen, not a distance-search reading.

    With no words <=4, colliding distinct triples are disjoint.
    Each weight-six support has ten unordered triple partitions.
    """
    cols = _l2_columns(rows, width)
    singles = set(cols)
    if 0 in singles:
        return 1, 0
    if len(singles) != width:
        return 2, 0
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles:
                return 3, 0
            if syndrome in pairs:
                return 4, 0
            pairs.add(syndrome)
    if not through_six:
        return 5, 0
    triples, collisions = {}, 0
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                syndrome = partial ^ cols[h]
                if syndrome in pairs:
                    return 5, 0
                old = triples.get(syndrome, 0)
                collisions += old
                triples[syndrome] = old + 1
    assert collisions % 10 == 0
    return (6 if collisions else 7), collisions // 10


def _l2_component(G, grid, side):
    """Extract only from the verified lift; the zero dummy is never emitted."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows], width


def _l2_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _l2_probe(rows, width, rng, passes, seed=0, own=None):
    """Information-set search returning an explicit original-coordinate word."""
    cols = _l2_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _l2_outside(word, own):
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
        rng.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            for j in range(len(kernel)):
                if i == j:
                    continue
                trial = word ^ kernel[j]
                if 0 < trial.bit_count() < word.bit_count():
                    word = trial
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
        if weight <= 5:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _l2_outside(best, own)
    return best


def _l2_signature(rows, width):
    """Conservative full, uncoloured Tanner-isomorphism invariant.

    Different signatures imply nonisomorphism. Equal signatures do not
    establish isomorphism and are rejected conservatively.
    """
    from collections import Counter
    from hashlib import sha256

    cols = _l2_columns(rows, width)
    incidence, shared = [], Counter()
    for col in cols:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(checks)
        for i, a in enumerate(checks):
            for b in checks[:i]:
                shared[(min(a, b), max(a, b))] += 1
    adjacency = [{} for _ in rows]
    for (a, b), multiplicity in shared.items():
        adjacency[a][b] = multiplicity
        adjacency[b][a] = multiplicity
    masks = [sum(1 << j for j in neighbors) for neighbors in adjacency]
    degrees = [row.bit_count() for row in rows]
    features = []
    for i, neighbors in enumerate(adjacency):
        histogram = Counter(
            (degrees[j], neighbors.get(j, 0),
             (masks[i] & masks[j]).bit_count())
            for j in range(len(rows)) if j != i)
        features.append((degrees[i], tuple(sorted(histogram.items()))))
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in checks))
                     for checks in incidence]
        qpalette = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for checks, feature in zip(incidence, qfeatures):
            for r in checks:
                profiles[r].append(qpalette[feature])
        refined = [
            (colors[i], tuple(sorted(profiles[i])),
             tuple(sorted((count, colors[j])
                          for j, count in adjacency[i].items())))
            for i in range(len(rows))]
        history.append((tuple(sorted(Counter(qfeatures).items())),
                        tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return sha256(repr((width, len(rows), history)).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _inactive_l2_25x38_portfolio(G, count):
    """Deterministic portfolio; cached publication data are immutable.

    Memoization changes only build cost. No slot can mutate the cached
    grids, and the input slot RNG does not affect another slot's choice.
    """
    assert G.q == 12
    q = G.q
    ap = (
        ((1, 1, 1, 3, 0), (1, 1, 1, 0, 3)),
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((0, 2, 2, 1, 1), (3, 1, 1, 1, 0)),
        ((1, 1, 2, 2, 0), (1, 1, 1, 1, 2)),
    )
    bp = (
        ((1, 1, 0, 0, 0, 2, 2, 0),
         (0, 1, 1, 1, 1, 0, 0, 2),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 0, 1, 1, 1, 1),
         (1, 1, 1, 1, 0, 0, 1, 1),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((1, 1, 1, 0, 0, 1, 2, 0),
         (0, 1, 0, 1, 1, 1, 0, 2),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 0, 1, 1, 2, 0),
         (1, 1, 1, 1, 0, 0, 1, 1),
         (0, 0, 1, 1, 1, 1, 0, 2)),
    )
    for a in ap:
        for b in bp:
            assert _no_dead_lines(a, b)
            assert all(sum(row) == 6 for row in a + b)
            assert _grid_weights(a, b) <= min(9, W_MAX)

    def element_order(g):
        power = G.e
        for value in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return value
        raise ValueError("invalid finite-group order")

    orders = {g: element_order(g) for g in G.elements}
    directions = {
        value: [g for g in G.elements if orders[g] == value]
        for value in (3, 4, 6)}
    triangles = []
    for g in G.elements:
        if g == G.e:
            continue
        for h in G.elements:
            if h in (G.e, g):
                continue
            relative = G.mul(G.inv(g), h)
            spectrum = tuple(sorted((orders[g], orders[h],
                                     orders[relative])))
            triangles.append((spectrum,
                              G.mul(g, h) != G.mul(h, g), g, h))

    def draw(profile, policy, local, trial):
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
                    # A sole binomial column uses order six rather than
                    # installing an order-four component witness.
                    column_degree = sum(r[j] for r in profile)
                    value = (6 if column_degree == 2 else
                             (4, 6)[(policy + i + j + trial) % 2])
                    relative = local.choice(directions[value])
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    if policy == 0 and (trial + i + j) % 2 == 0:
                        # A genuine normal-C3 annihilator prescription.
                        relatives = [G.e, (1, 0), (2, 0)]
                    else:
                        eligible = [
                            t for t in triangles
                            if (policy == 4
                                or (policy == 1 and t[1]
                                    and min(t[0]) >= 3)
                                or (policy == 2 and t[1]
                                    and 6 in t[0])
                                or (policy == 3 and t[0].count(4) >= 2)
                                or (policy == 0 and t[1]))]
                        if not eligible:
                            eligible = triangles
                        _, _, g, h = local.choice(eligible)
                        relatives = [G.e, g, h]
                    cell = [G.mul(anchor, g) for g in relatives]
                assert len(set(cell)) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def bank(side, profiles):
        entries, graph_keys = [], set()
        for card in range(20):
            profile, policy = profiles[card // 5], card % 5
            local = random.Random(
                BASE_SEED ^ (0x253812 + 7919 * card
                             + 104729 * int(side == "B")))
            raw, fillings = [], set()
            for trial in range(96):
                grid = draw(profile, policy, local, trial)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = _l2_component(G, grid, side)
                if _l2_floor(rows, width)[0] < 5:
                    continue
                raw.append((_l2_cycles(rows), trial, grid, rows, width))
            raw.sort(key=lambda r: r[:2])
            refined = []
            for cycles, trial, grid, rows, width in raw[:12]:
                floor, sixes = _l2_floor(rows, width, True)
                word = _l2_probe(rows, width, local, 4)
                if not word:
                    continue
                refined.append(dict(
                    grid=grid, rows=rows, width=width, word=word,
                    floor=floor, sixes=sixes, cycles=cycles,
                    card=card, trial=trial))
            refined.sort(key=lambda r: (
                r["floor"], min(r["word"].bit_count(), 8),
                -r["sixes"], -r["cycles"], -r["trial"]), reverse=True)
            for record in refined[:5]:
                record["word"] = _l2_probe(
                    record["rows"], record["width"], local, 12,
                    record["word"])
            refined = refined[:5]
            refined.sort(key=lambda r: (
                r["floor"], min(r["word"].bit_count(), 8),
                -r["sixes"], -r["cycles"], -r["trial"]), reverse=True)
            retained = 0
            for record in refined:
                signature = _l2_signature(record["rows"], record["width"])
                if signature in graph_keys:
                    continue
                graph_keys.add(signature)
                entries.append(record)
                retained += 1
                if retained == 3:
                    break
        if len(entries) < 5:
            raise ValueError("too few distinct screened constituents")
        return entries

    aa, bb = bank("A", ap), bank("B", bp)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            # Twenty deliberate cross-card prescriptions precede the
            # broader fallback pairing phase at comparable quality.
            prescribed = b["card"] == (7 * a["card"] + 3) % 20
            priority = (
                min(a["floor"], b["floor"]),
                min(a["word"].bit_count(), b["word"].bit_count(), 8),
                int(prescribed), -a["sixes"] - b["sixes"],
                -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda r: (r[0], -r[1], -r[2]), reverse=True)

    def reduce_word(word, stabilizers):
        for _ in range(2):
            starting = word
            ranked = sorted(
                stabilizers, key=lambda row: (starting ^ row).bit_count())
            for row in ranked:
                trial = starting ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            short = ranked[:24]
            for i, first in enumerate(short):
                for second in short[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def quantum_witness(a, b, hx, hz, n, local):
        readings = []
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _l2_basis(stabilizers)
            seeds = []
            for block in range(5 if kind == "X" else 8):
                if kind == "X":
                    word = b["word"] << (block * 8 * q)
                else:
                    word, source = 0, a["word"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 8 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if _l2_outside(word, own):
                    seeds.append(word)
            if not seeds:
                continue
            seed = min(seeds, key=int.bit_count)
            seed = reduce_word(seed, stabilizers)
            word = _l2_probe(checks, n, local, 2, seed, own)
            if word:
                word = reduce_word(word, stabilizers)
                assert word and _l2_outside(word, own)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                readings.append(word.bit_count())
        return min(readings) if readings else None

    results, graph_keys = [], set()
    local = random.Random(BASE_SEED ^ 0x3502538)
    wanted = count + max(20, count // 2)
    for priority, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(8))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert max(wx, wz) <= min(W_MAX, 9, 12)
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 552 and k > 0
                and 325 * n <= 1000 * k < 375 * n):
            continue
        signature = _l2_signature(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        # Final full-matrix screens include mixed-sector supports.
        if (_l2_floor(hx, n)[0] < 5
                or _l2_floor(hz, n)[0] < 5):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified shape portfolio failed CSS")
        reading = quantum_witness(a, b, hx, hz, n, local)
        if reading is None:
            continue
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
        frozen_A = tuple(tuple(tuple(cell) for cell in row) for row in A)
        frozen_B = tuple(tuple(tuple(cell) for cell in row) for row in B)
        # The cap prevents weakly searched, very large witness readings
        # from overwhelming the exact constituent information.
        score = k * min(reading, 8) ** 2 / n
        quality = (score, min(a["floor"], b["floor"]),
                   -a["sixes"] - b["sixes"],
                   -a["cycles"] - b["cycles"])
        results.append((quality, frozen_A, frozen_B, k, reading,
                        (a["card"], b["card"]),
                        (a["floor"], b["floor"]),
                        (a["sixes"], b["sixes"])))
        if len(results) >= wanted:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


@lru_cache(maxsize=1)
def _l2_portfolio(G, count):
    """Pure deterministic search, publishing only immutable complete grids.

    Exact ranks and all matrix screens use the verified scaffold lift.
    Component rank deficits are selection features, not predictions of k.
    """
    from itertools import permutations

    assert G.q == 6
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    cap = min(9, W_MAX)
    local_support = random.Random(BASE_SEED ^ 0x410386)

    def support_key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(len(profile[0]))))
            for order in permutations(range(len(profile))))

    # K4 edges followed by its four three-vertex faces.
    base = [[0] * ca for _ in range(ra)]
    edges = ((0, 1), (0, 2), (0, 3),
             (1, 2), (1, 3), (2, 3))
    for j, edge in enumerate(edges):
        for i in edge:
            base[i][j] = 1
    for omitted in range(ra):
        for i in range(ra):
            if i != omitted:
                base[i][6 + omitted] = 1

    aprofiles = [tuple(tuple(row) for row in base)]
    support_keys = {support_key(aprofiles[0])}
    current = [list(row) for row in base]
    for attempt in range(8192):
        D = [list(row) for row in current]
        u, v = local_support.sample(range(ra), 2)
        a, b = local_support.sample(range(ca), 2)
        if not D[u][a] or not D[v][b]:
            continue
        D[u][a] -= 1
        D[v][a] += 1
        D[v][b] -= 1
        D[u][b] += 1
        # Pure degree-two binomials on S3 immediately have a word <=3.
        # Keep those columns as genuine matching edges.
        if any(max(D[i][j] for i in range(ra)) > 1
               for j in range(6)):
            continue
        # Cubic columns may be full faces or split binomial endpoints,
        # but isolated trinomials are not this support family.
        if any(max(D[i][j] for i in range(ra)) > 2
               for j in range(6, ca)):
            continue
        current = D
        profile = tuple(tuple(row) for row in D)
        key = support_key(profile)
        if key in support_keys:
            continue
        support_keys.add(key)
        aprofiles.append(profile)
        if len(aprofiles) == 4:
            break
    if len(aprofiles) < 4:
        raise ValueError("tetrahedral support exchange catalogue exhausted")

    # Singleton at row 0; the repeated edge is between rows 1 and 2.
    # Putting the singleton on an endpoint of a repeated edge would
    # expose an elementary four-word grounded path.
    core = (
        (1, 1, 1, 0, 0),
        (0, 1, 0, 1, 1),
        (0, 0, 1, 1, 1),
    )
    cubic_cards = (
        ((1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((2, 1, 0), (0, 2, 1), (1, 0, 2)),
        ((1, 1, 1), (2, 1, 0), (0, 1, 2)),
        ((2, 0, 1), (0, 1, 2), (1, 2, 0)),
    )
    bprofiles = [
        tuple(tuple(core[i]) + tuple(column[i] for column in card)
              for i in range(rb))
        for card in cubic_cards
    ]
    for sa in aprofiles:
        for sb in bprofiles:
            assert all(sum(row) == 6 for row in sa + sb)
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) == 9 <= cap

    def order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid group element order")

    orders = {g: order(g) for g in G.elements}
    steps = {t: [g for g in G.elements if orders[g] == t]
             for t in (2, 3)}
    triangles = [
        (g, h) for g in G.elements for h in G.elements
        if g != G.e and h not in (G.e, g)
        and G.mul(g, h) != G.mul(h, g)
    ]

    def draw(profile, side, policy, local, trial):
        grid = []
        for i, row in enumerate(profile):
            cells = []
            for j, terms in enumerate(row):
                anchor = G.pick(local)
                if not terms:
                    cell = []
                elif terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    # Alternate genuine relative-order classes; do not
                    # obtain cards by applying one common automorphism.
                    wanted = (3 if policy in (0, 1)
                              else 2 if policy == 2
                              else (2, 3)[(trial + i + j + policy) % 2])
                    g = local.choice(steps[wanted])
                    cell = [anchor, G.mul(anchor, g)]
                else:
                    g, h = local.choice(triangles)
                    cell = [anchor, G.mul(anchor, g), G.mul(anchor, h)]
                assert len({G._key(g) for g in cell}) == terms
                cells.append(cell)
            grid.append(cells)

        if side == "B":
            # Normalize only the parallel matching bundle. Its relative
            # order three supplies six-cycles rather than four-cycles.
            grid[1][3], grid[2][3] = [G.e], [G.e]
            grid[1][4] = [G.e]
            grid[2][4] = [local.choice(steps[3])]
        return grid

    def bank(side, profiles):
        entries, graph_keys, fillings = [], set(), set()
        for card in range(20):
            profile, policy = profiles[card // 5], card % 5
            local = random.Random(
                BASE_SEED ^ (0x61038 + 104729 * card
                             + 1000003 * int(side == "B")))
            raw = []
            for trial in range(160):
                grid = draw(profile, side, policy, local, trial)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = _l2_component(G, grid, side)
                if _l2_floor(rows, width)[0] < 5:
                    continue
                rank = gf2_rank(rows)
                raw.append((rank, _l2_cycles(rows), trial,
                            grid, rows, width))
            # Refinement includes low-rank candidates, not only the
            # lowest-cycle full-rank samples.
            raw.sort(key=lambda item: item[:3])
            refined = []
            for rank, cycles, trial, grid, rows, width in raw[:28]:
                floor, sixes = _l2_floor(rows, width, True)
                word = _l2_probe(rows, width, local, 4)
                if not word:
                    continue
                refined.append(dict(
                    grid=grid, rows=rows, width=width, rank=rank,
                    cycles=cycles, trial=trial, card=card,
                    floor=floor, sixes=sixes, word=word))
            refined.sort(key=lambda r: (
                min(r["floor"], 6), -r["rank"],
                min(r["word"].bit_count(), 8),
                -r["sixes"], -r["cycles"]), reverse=True)
            for record in refined[:6]:
                record["word"] = _l2_probe(
                    record["rows"], record["width"], local, 12,
                    record["word"])
            retained = 0
            for record in refined[:6]:
                signature = _l2_signature(record["rows"], record["width"])
                if signature in graph_keys:
                    continue
                graph_keys.add(signature)
                entries.append(record)
                retained += 1
                if retained == 3:
                    break
        if len(entries) < 3:
            raise ValueError("too few invariant-distinct four-row constituents")
        return entries

    aa, bb = bank("A", aprofiles), bank("B", bprofiles)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (
                min(a["floor"], b["floor"], 6),
                -a["rank"] - b["rank"],
                min(a["word"].bit_count(), b["word"].bit_count(), 8),
                -a["sixes"] - b["sixes"],
                -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(reverse=True)

    n1 = ca * cb * q

    def mixed_five(rows, n):
        """Exact remaining five-word test after constituent exclusion.

        Pure sector-1 words split into independent constituent kernels.
        Every other five-support has a triple containing sector 2.
        The prior full kernel-four gate excludes overlapping collisions.
        """
        cols = _l2_columns(rows, n)
        pairs = set()
        for i, first in enumerate(cols):
            for j in range(i):
                pairs.add(first ^ cols[j])
        for i in range(n1, n):
            first = cols[i]
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    if (partial ^ cols[h]) in pairs:
                        return True
        return False

    def reduce_word(word, stabilizers):
        for _ in range(2):
            starting = word
            ranked = sorted(
                stabilizers, key=lambda row: (starting ^ row).bit_count())
            for row in ranked:
                trial = starting ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            for i, first in enumerate(ranked[:24]):
                for second in ranked[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def quantum_reading(a, b, hx, hz, n, local):
        readings = []
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _l2_basis(stabilizers)
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
                if _l2_outside(word, own):
                    seeds.append(word)
            seed = min(seeds, key=int.bit_count) if seeds else 0
            if seed:
                seed = reduce_word(seed, stabilizers)
            word = _l2_probe(checks, n, local, 2, seed, own)
            if not word:
                return None
            word = reduce_word(word, stabilizers)
            assert word and _l2_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            readings.append(word.bit_count())
        return min(readings)

    results, graph_keys = [], set()
    local = random.Random(BASE_SEED ^ 0x35041038)
    wanted = max(count, 5) + 12
    tested = 0
    for priority, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(rb))
                 for row in sa for j in range(cb))
        wz = max(sum(row) + sum(sa[i][j] for i in range(ra))
                 for row in sb for j in range(ca))
        assert max(wx, wz) <= cap
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 552 and 325*n <= 1000*k < 375*n):
            continue
        if (_l2_floor(hx, n)[0] < 5
                or _l2_floor(hz, n)[0] < 5):
            continue
        signature = _l2_signature(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        if not css_ok(hx, hz):
            raise ValueError("verified four-row portfolio failed CSS")

        complete_floor = 5
        if min(a["floor"], b["floor"]) >= 6:
            if not mixed_five(hx, n) and not mixed_five(hz, n):
                complete_floor = 6
        reading = quantum_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue
        assert reading >= complete_floor
        assert max(row.bit_count() for row in hx + hz) <= cap
        assert lifted_k(G, A, B) == k

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        # Exact six-clean codes precede weaker fallbacks. Within that
        # tier, measured rank surplus is the main improvement mechanism.
        score = k * min(reading, 8) ** 2 / n
        quality = (complete_floor, score, k,
                   -a["sixes"] - b["sixes"],
                   -a["cycles"] - b["cycles"])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["floor"], b["floor"]),
            (a["sixes"], b["sixes"])))
        tested += 1
        if len(results) >= wanted or tested >= 160:
            break
    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, invariant-separated complete design."""
    assert G.q == 6 and 0 <= idx < N_SLOTS
    portfolio = _l2_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct rank-surplus shape portfolio exhausted")
    quality, A, B, k, reading, cards, floors, sixes = portfolio[idx]
    return dict(A=A, B=B, screen_k=k, quality=quality,
                logical_witness_upper_bound=reading,
                design_cards=cards, constituent_kernel_floors=floors,
                constituent_six_word_counts=sixes)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent screens and the two-second proxy suffice to protect the Frobenius-bed finalists from weight-four logicals; slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: All-binomial cross-Cayley pairing alone supplies five inequivalent k=226 finalists: the measured parent supplied three, slot 3 collapsed to slot 0, and its other two finalists required k=222 trinomial fillings.
# REFUTED: The order-111 ownership implementation prevents all downstream duplicate collapse: its sixty-slot evaluation retained thirty-four distinct codes and twenty-six iso_dup slots.
# REFUTED: The order-12 support/kernel-six portfolio exceeds fitness 12.33 or supplies twenty inequivalent designs: sixty slots retained eleven classes, and all five measured codes had k=180,d30s=6, scoring 11.7390.
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
