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
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("On dihedral D26, order 26, A=(1,3), B=(2,7), one order-13 A binomial and Singer difference-set B matching columns should improve worse-basis ASC circuit fitness beyond 3.9894 by reducing extraction load and degree-two short-cycle opportunities.")
GROUP = ("Frozen L1: retain exactly D26 and its order-26 encoding. This bed achieved 3.9894; the same group's different (2,4)x(2,4) bed reached about 6.7, which establishes no group advantage here. Its C13 rotation subgroup admits the cyclic (13,4,1) difference set {0,1,3,9}, motivating a filling-only connection experiment.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=598, kb=260 and guaranteed rate 10/23=0.43478. The size factor 23 permits q<=26. Unequal components retain 78 variables/26 checks and 182 variables/52 checks. Require verified k=260, preserving zero k-kb. Parent weight-six witnesses occupied widespread sector-one Acol/Bcol blocks; failure enrichment was only 1.0-1.2x and no persistent Brow hotspot was established.")
FILLING = ("Twenty contracts cross four relative A geometries with five strengthened B architectures. A counts become (2,3,3); B rows total nine and columns have degrees (4,3,3,2,2,2,2). Maximum check weight remains twelve, while six-round CNOTs fall from 23868 to 22464. Initial matching tails use Singer voltages: their degree-two subgraph has Tanner girth twelve, although strengthened cells may introduce shorter component words. Additional harness batches change internal-edge or reflection-matching structure. Verified component actions screen rank and short kernels; rooted fingerprints and deterministic prefix reservation discard obvious repeated component pairs. Full exact-rate finalists receive common-stream six-round noisy-syndrome BP probes and independent holdouts before structural tie-breaks. Target slot6's 110/1250 Z and 121/1250 X failures, broad collective flips and late settlement. Probes omit ASC ordering and relay/LSD and are not measured circuit pL.")
FAMILY_NAME = "r045_D26_13x27_singer_light_temporalbp_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("D26 requires order 26")
    # x^13=y^2=1, y*x*y^-1=x^-1; elements are x^a*y^b.
    m = 13
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % 2)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % 2)

    def inv(g):
        sign = -1 if g[1] % 2 else 1
        return ((-sign * g[0]) % m, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return a + m * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Column term-count pairs. Every architecture has row totals (9,9),
# four degree-three columns and three degree-two columns.
_B_PROFILES = (
    ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (0,3), (2,1), (1,2), (1,1), (1,1), (1,1)),
    ((2,1), (2,1), (1,2), (1,2), (2,0), (0,2), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (2,0), (0,2), (1,1)),
)


def _component_rows(G, cells, side):
    """Use verified lifts, with a zero opposite side only as a probe."""
    if side == "A":
        return lifted_matrices(G, cells, [[[]]])[0]
    return lifted_matrices(G, [[[]]], cells)[1]


def _columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _kernel4(rows, width):
    """Exact weight-three/four counts after excluding weights one/two."""
    cols = _columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = {}
    three = four = boundary = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            three += int(value in singles)
            old = pairs.get(value, 0)
            four += old
            pairs[value] = old + 1
            w = value.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    assert three % 3 == 0 and four % 3 == 0
    return (three // 3, four // 3, boundary), cols


def _kernel56(cols):
    """Exact counts, called only when component weights <=4 are absent.

    Pair/triple matches count each weight-five support ten times;
    collisions of distinct triples count each weight-six support ten
    times. The <=4 exclusion makes these matching supports disjoint.
    """
    pairs = {}
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            pairs[value] = pairs.get(value, 0) + 1
    triples = {}
    five = six = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                five += pairs.get(value, 0)
                old = triples.get(value, 0)
                six += old
                triples[value] = old + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _four_cycles(rows):
    return sum(v * (v - 1) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Exact cycles using three checks and three distinct variables."""
    size = len(rows)
    overlaps = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i):
            overlaps[i][j] = overlaps[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(size):
        for j in range(i):
            x = overlaps[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlaps[j][h], overlaps[i][h]
                if y and z:
                    t = (common & rows[h]).bit_count()
                    total += x * y * z - t * (x + y + z) + 2 * t
    return total


def _rotation_geometries(m):
    """Unordered direction triples modulo actual rotation automorphisms."""
    from itertools import combinations

    def direction(x):
        x %= m
        return min(x, m - x)

    keys = set()
    for triple in combinations(range(1, (m + 1) // 2), 3):
        key = min(tuple(sorted(direction(u * d) for d in triple))
                  for u in range(1, m))
        keys.add(key)
    book = sorted(keys)
    assert len(book) == 4
    return book


def _component_score(short, cols, rows, four):
    # Short words take priority. A failed <=4 exclusion is not presented
    # as an exact weight-five/six spectrum.
    if short[0] or short[1]:
        five = six = 1 << 60
    else:
        five, six = _kernel56(cols)
    return (short[0], short[1], five, six,
            four, _six_cycles(rows), short[2])


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def NEW_CTX(rng, G, idx):
    """Pure structural catalogue, unused-pair reservation and BP holdouts."""
    import hashlib
    from collections import Counter

    q, m = G.q, G.q // 2
    design, batch = idx % 20, idx // 20
    assert q == 26 and m == 13
    sa = [[2, 3, 3]]
    heads = (
        ((2,2), (2,1), (1,2)),
        ((3,1), (1,2), (1,2)),
        ((3,1), (0,3), (2,1)),
        ((4,0), (1,2), (0,3)),
        ((2,2), (3,0), (0,3)),
    )
    tails = (
        ((1,1), (1,1), (1,1), (1,1)),
        ((2,0), (0,2), (1,1), (1,1)),
        ((1,1), (1,1), (1,1), (1,1)),
    )
    profiles = [head + tails[batch % 3] for head in heads]
    for profile in profiles:
        sb = [[c[i] for c in profile] for i in range(2)]
        assert tuple(map(sum, sb)) == (9, 9)
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 12

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def fingerprint(rows, width):
        """Rooted uncolored component invariant, not Tanner canonization.

        Include each check-row orbit. Equal keys are conservatively
        collapsed; different component pairs do not certify different
        complete LP codes, which the evaluator still checks.
        """
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, word in enumerate(_columns(rows, width)):
            while word:
                bit = word & -word
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                word ^= bit
        roots = []
        for root in range(0, nr, q):
            colors = [0] * nr + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(12):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            roots.append(digest.hexdigest())
        return tuple(sorted(roots))

    def draw_A(rr, model):
        # A common automorphism normalizes the binomial displacement.
        # The two remaining cells specify relative structures, not
        # independent translations of a saved complete code.
        def pure():
            u, v = rr.sample(range(1, m), 2)
            return [G.e, (u, 0), (v, 0)]

        def mixed(d, phase):
            return [G.e, (d, 0), (phase, 1)]

        if model == 0:
            d, e = rr.sample(range(2, 7), 2)
            cells = [mixed(d, 0), mixed(e, rr.randrange(m))]
        elif model == 1:
            cells = [pure(), mixed(rr.randrange(2, 7), 0)]
        elif model == 2:
            cells = [pure(), pure()]
        else:
            cells = [mixed(1, 0),
                     mixed(rr.randrange(2, 7), rr.randrange(m))]
        return [[[G.e, (1, 0)]] + cells]

    def draw_B(rr, architecture):
        profile = profiles[architecture]
        B = [[[] for _ in range(7)] for _ in range(2)]
        # For (1,1), t=b0*b1^-1 is the verified right-component
        # connection. The Singer set's twelve ordered differences
        # exhaust C13\{0}, excluding check-graph four-cycles.
        if batch % 3 == 0:
            holonomies = [(t, 0) for t in (0, 1, 3, 9)]
        elif batch % 3 == 1:
            holonomies = [(0, 0), (rr.randrange(1, m), 1)]
        else:
            # A genuinely different matching subgraph: relative
            # reflections can have order two, unlike Singer tails.
            a, b = rr.sample(range(m), 2)
            holonomies = [G.e, (1, 0), (a, 1), (b, 1)]
        matching = 0
        for j, column in enumerate(profile):
            if column == (1, 1):
                t = holonomies[matching]
                matching += 1
                B[0][j], B[1][j] = [G.e], [G.inv(t)]
                continue
            for i, count in enumerate(column):
                if not count:
                    continue
                base = G.pick(rr)
                if count == 1:
                    relative = [G.e]
                elif count == 2:
                    relative = [G.e, (rr.randrange(1, m), 0)]
                elif count == 3:
                    if (i + j + batch) % 3 == 0:
                        u, v = rr.sample(range(1, m), 2)
                        relative = [G.e, (u, 0), (v, 0)]
                    else:
                        relative = [G.e, (rr.randrange(1, m), 0),
                                    (rr.randrange(m), 1)]
                else:
                    u = rr.randrange(1, m)
                    v, w = rr.sample(range(m), 2)
                    relative = [G.e, (u, 0), (v, 1), (w, 1)]
                cell = [G.mul(base, g) for g in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    def catalogue(side, card):
        # Rebuild the same banks in every call: no global cache, mutation
        # of saved codes, or dependence on invocation order.
        rr = random.Random(260598731 + 1009 * card
                           + (0 if side == "A" else 65537 * (batch + 1)))
        width = (3 if side == "A" else 7) * q
        rank = (1 if side == "A" else 2) * q
        pool, seen = [], set()
        trials = 192 if side == "A" else 320
        for trial in range(trials):
            grid = (draw_A(rr, card) if side == "A"
                    else draw_B(rr, card))
            identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for row in grid for cell in row)
            if identity in seen:
                continue
            seen.add(identity)
            rows = _component_rows(G, grid, side)
            if gf2_rank(rows) != rank:
                continue
            result = _kernel4(rows, width)
            if result is None:
                continue
            short, cols = result
            four = _four_cycles(rows)
            coarse = (short[0], short[1], four, short[2])
            pool.append((coarse, trial, grid, rows, cols, short, four))
            pool.sort(key=lambda item: item[:2])
            del pool[16:]

        book = {}
        for coarse, trial, grid, rows, cols, short, four in pool:
            identity = fingerprint(rows, width)
            if identity in book:
                continue
            spectrum = _component_score(short, cols, rows, four)
            degree_two = [c for c in cols if c.bit_count() == 2]
            sparse6 = 1 << 60
            if not (short[0] or short[1]):
                _, sparse6 = _kernel56(degree_two)
            key = spectrum[:4] + (sparse6,) + spectrum[4:]
            book[identity] = {
                "key": key, "id": identity, "grid": clone(grid),
                "spectrum": spectrum, "sparse6": sparse6,
            }
        entries = sorted(book.values(), key=lambda item: (
            item["key"], item["id"]))
        if not entries:
            raise ValueError("no full-rank distinct component in contract")
        return entries[:6]

    abooks = [catalogue("A", card) for card in range(4)]
    bbooks = [catalogue("B", card) for card in range(5)]
    catalogues = []
    for model in range(4):
        for architecture in range(5):
            choices = []
            for a in abooks[model]:
                for b in bbooks[architecture]:
                    ac, bc = a["spectrum"], b["spectrum"]
                    protection = (
                        max(ac[0], bc[0]), ac[0] + bc[0],
                        max(ac[1], bc[1]), ac[1] + bc[1],
                    )
                    # Component spectra are shortlist statistics, not
                    # an exact census of nontrivial quantum logicals.
                    geometry = (
                        max(7*ac[2], 3*bc[2]), 7*ac[2] + 3*bc[2],
                        b["sparse6"],
                        max(7*ac[3], 3*bc[3]), 7*ac[3] + 3*bc[3],
                        ac[4] + bc[4], ac[5] + bc[5],
                    )
                    identity = (a["id"], b["id"])
                    choices.append((protection + geometry, identity, a, b))
            choices.sort(key=lambda item: item[:2])
            catalogues.append(choices)

    # Reserve all finalists of every preceding card, not only the one
    # its decoder probe would select. Component identities are uncolored.
    reserved = set()
    chosen = None
    actual_card = design
    for slot in range(design + 1):
        card = slot
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            alternatives = [
                (item[0], item[1], c, item)
                for c, book in enumerate(catalogues)
                for item in book if item[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("unused component-pair catalogue exhausted")
            card = min(alternatives, key=lambda item: item[:3])[2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:3]
        reserved.update(item[1] for item in chosen)
        actual_card = card

    options = []
    for number, (coarse, identity, a, b) in enumerate(chosen):
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 598 or k != 260:
            continue
        rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
        if rx is None or rz is None:
            continue
        protection = coarse[:4] + (
            max(rx[0], rz[0]), rx[0] + rz[0],
            max(rx[1], rz[1]), rx[1] + rz[1],
        )
        options.append((protection, coarse[4:], number, identity,
                        A, B, hx, hz, a, b))
    if not options:
        raise ValueError("no verified exact-rate joint finalist")
    options.sort(key=lambda item: item[:4])

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round noisy-syndrome BP surrogate, NOT ASC circuit pL.

        Data increments and measurement errors are decoded jointly,
        with a perfect closing syndrome. Approximate spread pairs omit
        ASC ordering; relay BP and LSD are absent.
        """
        import numpy as np

        def neighbors(rows):
            result = []
            for word in rows:
                row = []
                while word:
                    bit = word & -word
                    row.append(bit.bit_length() - 1)
                    word ^= bit
                result.append(row)
            return result

        ns, spread = neighbors(checks), neighbors(stabilizers)
        nc, rounds = len(ns), 6
        variables = rounds * (n + nc)
        dummy = variables
        detectors = (rounds + 1) * nc
        dc = max(map(len, ns)) + 2
        incidence = np.full((detectors, dc), dummy, dtype=np.int32)
        for t in range(rounds + 1):
            for r, row in enumerate(ns):
                position = 0
                if t < rounds:
                    incidence[t*nc+r, :len(row)] = [t*n+c for c in row]
                    position = len(row)
                    incidence[t*nc+r, position] = rounds*n+t*nc+r
                    position += 1
                if t:
                    incidence[t*nc+r, position] = rounds*n+(t-1)*nc+r
        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        ve = np.full((variables + 1, int(degree.max())),
                     edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            ve[c, used[c]] = edge
            used[c] += 1

        total_degree = np.asarray([
            x.bit_count() + z.bit_count()
            for x, z in zip(_columns(checks, n),
                            _columns(stabilizers, n))
        ], dtype=float)
        data_rate = 0.002 * (1 + 2*total_degree/3)
        measure_rate = np.asarray([
            0.002 * (2 + len(row)) * 2/3 for row in ns])
        base = np.concatenate((np.tile(data_rate, rounds),
                               np.tile(measure_rate, rounds)))
        levels = np.asarray((0.8, 1.1, 1.4))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None] * base[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables + 1), dtype=bool)
        error[:, :variables] = (
            channel.random((shots, variables)) < rates)
        for t in range(rounds):
            chosen_checks = channel.integers(len(spread), size=shots)
            active = channel.random(shots) < 0.25
            for shot in np.flatnonzero(active):
                row = spread[int(chosen_checks[shot])]
                u, v = channel.choice(len(row), 2, replace=False)
                error[shot, t*n+row[int(u)]] ^= True
                error[shot, t*n+row[int(v)]] ^= True

        target = (error[:, flat].reshape(
            shots, detectors, dc).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.full((shots, variables + 1), 30.0)
        prior[:, :variables] = np.log((1-rates)/rates)
        messages = prior[:, flat].copy()
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)

        pivots = {}
        for word in stabilizers:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        failures = effort = 0
        positions = np.arange(dc)[None, None, :]
        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, detectors, dc)
            signs = np.where(shaped < 0, -1.0, 1.0)
            magnitude = np.where(
                valid.reshape(detectors, dc)[None, :, :],
                np.abs(shaped), np.inf)
            at = np.argmin(magnitude, axis=2)
            smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
            size = np.where(
                positions == at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            outgoing = (
                0.80 * np.minimum(size, 30.0) * signs
                * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges + 1))
            padded[:, :edges] = outgoing
            posterior = np.clip(prior + padded[:, ve].sum(axis=2),
                                -50.0, 50.0)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, detectors, dc).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = decoded_final[shot] ^ true_final[shot]
                word = 0
                for c in np.flatnonzero(residual):
                    word |= 1 << int(c)
                failures += int(nontrivial(word))
                effort += iteration
            keep = ~settled
            if not np.any(keep):
                return failures, 0, effort
            messages = np.clip(
                0.30*messages
                + 0.70*(posterior[:, flat] - outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return failures + unresolved, unresolved, effort + 33*unresolved

    def quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2])

    # Compare only equally short-protected full matrices. Do not let
    # an optimistic small probe buy acceptance of a shorter kernel.
    floor = options[0][0]
    eligible = [item for item in options if item[0] == floor]
    seed = 260598193 + 1009*idx
    probed = []
    for option in eligible:
        protection, geometry, number, identity, A, B, hx, hz, a, b = option
        x = temporal_probe(hz, hx, 598, seed, 16)
        z = temporal_probe(hx, hz, 598, seed + 1, 16)
        key = quality(x, z) + geometry + (number,)
        probed.append((key, number, option, x, z))
    probed.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, option, cx, cz in probed[:2]:
        protection, geometry, _, identity, A, B, hx, hz, a, b = option
        x = temporal_probe(hz, hx, 598, seed + 1000003, 32)
        z = temporal_probe(hx, hz, 598, seed + 1000004, 32)
        total_x = tuple(u + v for u, v in zip(cx, x))
        total_z = tuple(u + v for u, v in zip(cz, z))
        key = protection + quality(total_x, total_z) + geometry + (
            number, identity)
        finalists.append((key, option, total_x, total_z))
    finalists.sort(key=lambda item: item[0])
    key, option, px, pz = finalists[0]
    _, _, _, identity, A, B, hx, hz, a, b = option
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 260
    assert tuple(map(len, A[0])) == (2, 3, 3)
    actual_sb = [[len(cell) for cell in row] for row in B]
    assert tuple(map(sum, actual_sb)) == (9, 9)
    assert sorted(map(sum, zip(*actual_sb))) == [2, 2, 2, 2, 3, 3, 4]
    assert _no_dead_lines(sa, actual_sb)
    assert _grid_weights(sa, actual_sb) == 12
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 22464
    assert 0.425 <= 260 / 598 < 0.475
    return {"A": A, "B": B, "k": 260,
            "design": design, "actual_card": actual_card, "batch": batch,
            "component_profiles": (a["spectrum"], b["spectrum"]),
            "degree_two_weight_six": b["sparse6"],
            "component_identity": identity,
            "temporal_bp_surrogate": (px, pz),
            "cnots": 22464, "ambiguity_score": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree A trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894, slot6 failed 110/1250 Z and 121/1250 X shots, all five measured codes retained abundant sector-one weight-six witnesses, and failures collectively affected all 260 logicals.
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
