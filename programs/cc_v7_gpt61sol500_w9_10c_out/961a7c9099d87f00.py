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
CONCEPT = ("Quaternion Q8, order 8, A=B=(3,7): a long-cycle three-edge core reinforced by four cubic columns should supply five measured distance-eight readings in cell 0.300, exceeding fitness 17.65 at k=128; this is falsifiable, not a quantum-distance certificate.")
GROUP = ("Jump from C3 semidirect C4 of order twelve to Q8. The presentation x^4=1,y^2=x^2,y*x*y^-1=x^-1 is not the parent's split semidirect product. Three noncentral cyclic subgroups provide noncommuting order-four voltages. The related order-eight beds do not establish performance on this untried equal-(3,7) bed; selection uses actual verified matrices.")
SHAPES = ("Equal (3,7) constituents have 56 variables and 24 checks. Size factor 58 permits q<=10; fixed Q8 gives n=464,kb=128 and guaranteed rate 16/58=0.275862, in cell 0.300. This cell's frontier is 15.207, below the neighbouring 0.250 frontier. Parent measured weight-six minima occupied sector-1 Acol/Bcol blocks broadly. Replace both asymmetric constituents and screen complete matrices, including mixed sectors. Verified ranks retain [0.275,0.325).")
FILLING = ("Ten canonical six-term-row support profiles crossed with two relative-voltage policies produce independently optimized left/right banks. Columns have degrees two or three, so BOTH CSS maxima are nine, also below twelve. The seed profile has a three-edge monomial core and four full cubic columns; additional profiles redistribute cubic terms rather than relabeling that seed. Exact constituent exclusion through six precedes equal-budget witness refinement. Full verified ranks, CSS, kernel-four rejection, nontrivial embedded witnesses and mixed-sector probes gate publication. An immutable portfolio rejects equal intrinsic uncoloured Tanner signatures. Distance-eight survival and unseen archive novelty remain hypotheses.")
FAMILY_NAME = "r030_quaternion8_37x37_core_cubic_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 has order eight")
    # (a,b)=x^a*y^b, x^4=1, y^2=x^2, y*x*y^-1=x^-1.
    # The cocycle 2*b*d distinguishes Q8 from the split dihedral law.
    els = [(a, b) for b in range(2) for a in range(4)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 2 * (b // 2)) % 4, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 2*b*d) % 4,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 4, 0) if b == 0 else ((a + 2) % 4, 1)

    def idx(g):
        a, b = red(g)
        return 4*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 7)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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
def _l2_portfolio(G, count):
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


def _q8_supports():
    """Canonical count designs: three degree-two and four cubic columns."""
    from itertools import permutations, combinations_with_replacement

    core = ((1, 0, 1), (1, 1, 0), (0, 1, 1))
    cubic = sorted(set(permutations((2, 1, 0)))
                   | set(permutations((3, 0, 0)))
                   | {(1, 1, 1)})

    def grid(cols):
        return tuple(tuple(c[i] for c in cols) for i in range(3))

    def key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(7)))
            for order in permutations(range(3)))

    seed = grid(core + ((1, 1, 1),) * 4)
    result, used = [seed], {key(seed)}
    candidates = []
    for extras in combinations_with_replacement(cubic, 4):
        if any(sum(c[i] for c in extras) != 4 for i in range(3)):
            continue
        profile = grid(core + extras)
        signature = key(profile)
        if signature in used:
            continue
        used.add(signature)
        # Interleave split cubic columns and isolated trinomials.
        split = sum(max(c) == 2 for c in extras)
        isolated = sum(max(c) == 3 for c in extras)
        candidates.append((isolated, -split, signature, profile))
    candidates.sort()
    result.extend(item[3] for item in candidates[:9])
    return tuple(result)


def _q8_core_girth(rows, width):
    """Exact girth of the degree-two subgraph of a verified component."""
    cols = _l2_columns(rows, width)
    adjacency = [[] for _ in rows]
    for edge, col in enumerate(cols):
        if col.bit_count() != 2:
            continue
        low = col & -col
        a = low.bit_length() - 1
        b = (col ^ low).bit_length() - 1
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    best = len(rows) + 1
    for root in range(len(rows)):
        distance = [-1] * len(rows)
        parent_edge = [-1] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if distance[b] < 0:
                    distance[b] = distance[a] + 1
                    parent_edge[b] = edge
                    queue.append(b)
                elif edge != parent_edge[a] and edge != parent_edge[b]:
                    best = min(best, distance[a] + distance[b] + 1)
    return best


@lru_cache(maxsize=1)
def _q8_portfolio(G, count):
    """Deterministic immutable portfolio; no cross-slot mutable choices.

    Short exclusions are exact. Larger witness readings are upper bounds.
    Unequal intrinsic Tanner signatures separate within-program designs;
    the evaluator's archive remains authoritative for historical novelty.
    """
    from itertools import combinations

    assert G.q == 8
    q, width, e = G.q, 7 * G.q, G.e
    cap = min(9, 12, W_MAX)
    profiles = _q8_supports()
    for profile in profiles:
        assert _no_dead_lines(profile, profile)
        assert all(sum(row) == 6 for row in profile)
        assert all(sum(row[j] for row in profile) in (2, 3)
                   for j in range(7))
        assert _grid_weights(profile, profile) == 9 <= cap

    def order(g):
        value = e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return t
        raise ValueError("invalid quaternion element order")

    orders = {g: order(g) for g in G.elements}
    fourths = [g for g in G.elements if orders[g] == 4]
    triangle_pools = [[], []]
    for g, h in combinations([g for g in G.elements if g != e], 2):
        noncommuting = G.mul(g, h) != G.mul(h, g)
        triangle_pools[int(noncommuting)].append((g, h))

    def draw(profile, policy, local):
        C = [[[] for _ in range(7)] for _ in range(3)]
        for j in range(7):
            for i in range(3):
                terms = profile[i][j]
                if not terms:
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    cell = [anchor, G.mul(anchor, local.choice(fourths))]
                else:
                    g, h = local.choice(triangle_pools[policy])
                    cell = [anchor, G.mul(anchor, g), G.mul(anchor, h)]
                assert len(set(cell)) == terms
                C[i][j] = cell

        # Parallel matching bundles receive genuinely different ratios.
        # This is only a proposal constraint; verified syndromes screen it.
        for u in range(3):
            for v in range(u):
                bundle = [
                    j for j in range(7)
                    if profile[u][j] == profile[v][j] == 1
                    and sum(row[j] for row in profile) == 2]
                if len(bundle) <= 1:
                    continue
                for _ in range(32):
                    labels = local.sample(G.elements, len(bundle))
                    if all(orders[G.mul(G.inv(a), b)] == 4
                           for a, b in combinations(labels, 2)):
                        for j, label in zip(bundle, labels):
                            C[v][j], C[u][j] = [e], [label]
                        break
        return C

    def bank(side):
        entries, invariant_counts = [], {}
        for card in range(20):
            profile = profiles[(card // 2) % len(profiles)]
            policy = card % 2
            local = random.Random(
                BASE_SEED * 10007 + 0x837307
                + 65537 * card + 104729 * int(side == "B"))
            raw, fillings = [], set()
            for trial in range(192):
                C = draw(profile, policy, local)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in C for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, extent = _l2_component(G, C, side)
                assert extent == width
                if gf2_rank(rows) != 3*q:
                    continue
                if _l2_floor(rows, width)[0] < 5:
                    continue
                girth = _q8_core_girth(rows, width)
                cycles = _l2_cycles(rows)
                raw.append((cycles, -girth, trial, C, rows))
            raw.sort(key=lambda r: r[:3])

            refined = []
            for cycles, ngirth, trial, C, rows in raw[:24]:
                floor, sixes = _l2_floor(rows, width, True)
                word = _l2_probe(rows, width, local, 6)
                if not word:
                    continue
                refined.append(dict(
                    grid=C, rows=rows, word=word, floor=floor,
                    sixes=sixes, cycles=cycles, girth=-ngirth,
                    card=card, trial=trial))
            refined.sort(key=lambda r: (
                r["floor"], min(r["word"].bit_count(), 10),
                -r["sixes"], -r["cycles"], r["girth"]),
                reverse=True)

            finalists = refined[:6]
            for record in finalists:
                record["word"] = _l2_probe(
                    record["rows"], width, local, 32, record["word"])
            finalists.sort(key=lambda r: (
                min(r["word"].bit_count(), 10), r["floor"],
                -r["sixes"], -r["cycles"]), reverse=True)
            kept = 0
            for record in finalists:
                signature = _l2_signature(record["rows"], width)
                # Two realizations allow genuine relative LP orientations.
                if invariant_counts.get(signature, 0) >= 2:
                    continue
                invariant_counts[signature] = (
                    invariant_counts.get(signature, 0) + 1)
                entries.append(record)
                kept += 1
                if kept >= 3:
                    break
        return entries

    aa, bb = bank("A"), bank("B")
    if not aa or not bb:
        return ()
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["word"].bit_count(), b["word"].bit_count())
            priority = (
                min(weak, 10), min(a["floor"], b["floor"]),
                int(a["card"] != b["card"]),
                -a["sixes"] - b["sixes"],
                -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda r: (r[0], -r[1], -r[2]), reverse=True)

    def descent(word, stabilizers):
        for _ in range(3):
            starting = word
            ranked = sorted(
                stabilizers, key=lambda row: (starting ^ row).bit_count())
            for row in ranked:
                trial = starting ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            for i, first in enumerate(ranked[:32]):
                for second in ranked[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def logical_reading(a, b, hx, hz, n, local):
        best = n + 1
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _l2_basis(stabilizers)
            seeds = []
            for block in range(7):
                if kind == "X":
                    word = b["word"] << (block * width)
                else:
                    word, source = 0, a["word"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j*7 + block)*q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if _l2_outside(word, own):
                    seeds.append(word)
            if not seeds:
                continue
            seed = descent(min(seeds, key=int.bit_count), stabilizers)
            # Complete information sets include both LP sectors.
            word = _l2_probe(checks, n, local, 4, seed, own)
            word = descent(word, stabilizers)
            assert word and _l2_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            best = min(best, word.bit_count())
        return best if best <= n else None

    results, graph_keys = [], set()
    tested = 0
    wanted = count + max(12, count // 2)
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(7))
        wz = max(sum(row) + sum(sa[i][j] for i in range(3))
                 for row in sb for j in range(7))
        assert wx == wz == 9 <= cap
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 464 and k > 0
                and 275*n <= 1000*k < 325*n):
            continue
        if (_l2_floor(hx, n)[0] < 5
                or _l2_floor(hz, n)[0] < 5):
            continue
        signature = _l2_signature(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        assert css_ok(hx, hz)
        local = random.Random(
            BASE_SEED ^ (0x837C00 + 65537*ai + 257*bi))
        reading = logical_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) == 9
        quality = (k * min(reading, 10)**2 / n,
                   min(a["floor"], b["floor"]),
                   -a["sixes"] - b["sixes"], preliminary)

        def freeze(C):
            return tuple(tuple(tuple(cell) for cell in row) for row in C)

        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (a["floor"], b["floor"]),
            (a["sixes"], b["sixes"])))
        if len(results) >= wanted or tested >= max(160, 4*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure publication of an invariant-separated quaternion design."""
    assert G.q == 8 and 0 <= idx < N_SLOTS
    portfolio = _q8_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("screened quaternion target-rate portfolio exhausted")
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
# REFUTED: The order-12 asymmetric support/kernel-six portfolio exceeds fitness 12.33 and supplies twenty evaluator-distinct designs: its measured fitness was 11.739, all five finalists had d30s=6, and sixty slots retained eleven classes.
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
