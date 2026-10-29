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
CONCEPT = ("On dihedral D26, order 26, shapes A=(1,3), B=(2,7), an eight-term A and rotation-only Sidon degree-two B connections, with noncommutation confined to strengthened cells, should improve worse-basis circuit fitness above 3.9894 by reducing extraction exposure without accepting involution-induced short cycles.")
GROUP = ("Frozen L1: retain exactly D26, order 26, its presentation and encoding. The order-13 rotation subgroup supports long binomial cycles and Sidon connection sets. Parent fitness 3.9894 and diffuse collective failures motivate a filling experiment, not a group-defect claim. Related C3:C8 fillings scored about 4.12, but those readings do not isolate a causal gate-count advantage.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=598, kb=260 and guaranteed rate 10/23=0.43478; the size factor 23 allows q<=26, exactly the fixed lift. Unequal components have 78 variables/26 checks and 182 variables/52 checks. Require verified k=260, retaining zero k-kb. Parent weight-six sector-one witnesses span many Acol/Bcol blocks; enrichment is only 1.0-1.2x and no persistent Brow hotspot is established.")
FILLING = ("Twenty contracts cross four normalized binomial/trinomial A direction briefs with five B support architectures. A counts are (2,3,3). Degree-two matching holonomies stay in the rotation subgroup: three-link tails use Sidon sets and four-link tails use the (13,4,1) difference set {0,1,3,9}; internal doubletons also use rotations. Strengthened cells introduce noncommutation. Helper-derived component searches compare short kernels and exact five/six counts when their preconditions hold. Complete verified lifts require k=260 and screen both syndrome ambiguity profiles. Independent two-interval noisy-syndrome BP holdouts lead selection after short-word protection; suffix-fault and degree-weighted channels omit ASC scheduling and relay/LSD and are not circuit fitness. Both weight sums remain <=12; six-round CNOTs are 21840 or 22464 instead of 23868. Target the parent's 110/1250 Z and 121/1250 X failures in slot 6, widespread logical flips, and extensive late X settlement rather than one unsupported block repair.")
FAMILY_NAME = "r045_D26_13x27_gatelean_sidon_holdout_v2"

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
    """Gate-lean direction cards with Sidon tails and independent holdouts."""
    import math
    from itertools import combinations

    q, m = G.q, G.q // 2
    assert q == 26 and m == 13
    design, batch = idx % 20, idx // 20
    geometry, architecture = divmod(design, 5)

    # The binomial displacement is normalized to one. Consequently,
    # these are relative direction contracts, not scale-orbit copies.
    direction_cards = (
        ((2,3), (2,4), (2,5), (3,4)),
        ((2,6), (3,5), (3,6), (4,5)),
        ((4,6), (5,6), (2,2), (3,3)),
    )
    steps = direction_cards[batch % 3][geometry]

    profiles = (
        ((2,1), (2,1), (1,2), (1,2),
         (1,1), (1,1), (1,1)),
        ((3,0), (1,2), (1,2), (1,2),
         (1,1), (1,1), (1,1)),
        ((2,1), (1,2), (1,2),
         (1,1), (1,1), (1,1), (1,1)),
        ((2,1), (2,1), (0,3),
         (2,0), (0,2), (1,1), (1,1)),
        ((2,2), (2,1), (1,2),
         (1,1), (1,1), (1,1), (1,1)),
    )
    profile = profiles[architecture]
    sa = [[2, 3, 3]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert max(map(sum, sb)) <= 9
    assert all(sum(p) >= 2 for p in profile)
    expected_cnots = 6*q*(9*8 + 4*sum(map(sum, sb)))
    assert expected_cnots in (21840, 22464)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Compose only singleton blocks supplied by the verified helper.
    # These zero-partner probes are never submitted as candidate codes.
    mask = (1 << q) - 1
    probe = [[[g] for g in G.elements]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {"A": [], "B": []}
    for t in range(q):
        actions["A"].append(tuple((row >> (t*q)) & mask for row in ax))
        actions["B"].append(tuple((row >> (t*q)) & mask for row in bz))

    def component(grid, side):
        rows = [0] * (len(grid)*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        rows[i*q+r] ^= word << (j*q)
        return rows

    def assess(grid, side):
        width = len(grid[0])*q
        rows = component(grid, side)
        if gf2_rank(rows) != len(grid)*q:
            return None
        result = _kernel4(rows, width)
        if result is None:
            return None
        short, cols = result
        four = _four_cycles(rows)
        return rows, cols, short, four

    def canonical_A(grid):
        # Right cell translations, actual D26 automorphisms, and
        # component-column permutations remove obvious proposal copies.
        # This is not complete full-code Tanner canonization.
        images = []
        for u in range(1, m):
            for v in range(m):
                normalized = []
                for cell in grid[0]:
                    image = [((u*a + v*b) % m, b) for a, b in cell]
                    normalized.append(min(
                        tuple(sorted(G.idx(G.mul(g, G.inv(origin)))
                                     for g in image))
                        for origin in image))
                images.append(tuple(sorted(normalized)))
        return min(images)

    apool, aseen = [], set()
    for phase in range(m):
        A = [[[G.e, (1,0)],
              [G.e, (steps[0],0), (0,1)],
              [G.e, (steps[1],0), (phase,1)]]]
        state = assess(A, "A")
        if state is None:
            continue
        identity = canonical_A(A)
        if identity in aseen:
            continue
        aseen.add(identity)
        rows, cols, short, four = state
        spectrum = _component_score(short, cols, rows, four)
        key = spectrum[:4] + (four, spectrum[6], spectrum[5])
        apool.append((key, phase, A, rows, spectrum))
    apool.sort(key=lambda item: item[:2])
    apool = apool[:3]
    if not apool:
        raise ValueError("no full-rank gate-lean A direction card")

    # All ordered differences in this set are distinct modulo thirteen.
    singer = (0, 1, 3, 9)
    differences = [(a-b) % m for a in singer for b in singer if a != b]
    assert len(set(differences)) == 12

    # Three-link tails likewise exclude repeated ordered differences.
    sidon3 = [
        (0, a, b) for a, b in combinations(range(1, m), 2)
        if len({(x-y) % m for x in (0,a,b) for y in (0,a,b)
                if x != y}) == 6
    ]
    assert sidon3
    matching_sites = [j for j, p in enumerate(profile) if p == (1,1)]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        cross_count = len(matching_sites)
        if cross_count == 4:
            scale = rng.randrange(1, m)
            holonomies = [(scale*t) % m for t in singer]
        elif cross_count == 3:
            holonomies = list(rng.choice(sidon3))
        elif cross_count == 2:
            holonomies = [0, rng.randrange(1, m)]
        else:
            raise ValueError("unsupported Sidon matching contract")
        # Permuting these links changes their coupling to the head,
        # rather than transforming a stored complete code.
        rng.shuffle(holonomies)
        paired = dict(zip(matching_sites, holonomies))

        for j, column in enumerate(profile):
            if j in paired:
                t = (paired[j], 0)
                B[0][j], B[1][j] = [G.e], [G.inv(t)]
                continue
            for i, count in enumerate(column):
                if count == 0:
                    continue
                base = G.pick(rng)
                if count == 1:
                    cell = [base]
                elif count == 2:
                    # A reflection ratio is involutory. In a pure
                    # doubleton it gives duplicate component columns;
                    # between matching columns it gives a four-cycle.
                    delta = (rng.randrange(1, m), 0)
                    cell = [base, G.mul(base, delta)]
                elif count == 3:
                    # Strengthened variables carry the noncommuting
                    # connections omitted from the degree-two tail.
                    delta = (rng.randrange(1, m), 0)
                    reflection = (rng.randrange(m), 1)
                    cell = [base, G.mul(base, delta),
                            G.mul(base, reflection)]
                else:
                    raise ValueError("unsupported strengthened cell")
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    bpool, bseen = [], set()
    serial = 0

    def remember_B(B, state):
        nonlocal serial
        if state is None:
            return
        identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                         for row in B for cell in row)
        if identity in bseen:
            return
        bseen.add(identity)
        rows, cols, short, four = state
        coarse = (short[0], short[1], four, short[2])
        serial += 1
        bpool.append((coarse, serial, clone(B), rows, cols, short, four))
        bpool.sort(key=lambda item: item[:2])
        del bpool[6:]

    for _ in range(384):
        B = draw_B()
        remember_B(B, assess(B, "B"))

    # Whole-column mutations preserve term budgets and all tail contracts.
    if bpool:
        for restart in range(2):
            B = clone(bpool[min(restart, len(bpool)-1)][2])
            state = assess(B, "B")
            for step in range(160):
                proposal = clone(B)
                fresh = draw_B()
                # Matching tails must be changed jointly to preserve
                # their Sidon property. Other columns change singly.
                if rng.randrange(4) == 0:
                    for j in matching_sites:
                        proposal[0][j], proposal[1][j] = (
                            fresh[0][j], fresh[1][j])
                else:
                    sites = [j for j in range(7) if j not in matching_sites]
                    j = rng.choice(sites)
                    proposal[0][j], proposal[1][j] = (
                        fresh[0][j], fresh[1][j])
                proposed = assess(proposal, "B")
                if proposed is None:
                    continue

                def energy(s):
                    _, _, short, four = s
                    return (500*short[0] + 80*short[1]
                            + 5*four + short[2]/16)/q

                delta = energy(proposed) - energy(state)
                temperature = 6*(0.976**step) + 0.05
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    B, state = proposal, proposed
                    if step % 16 == 0:
                        remember_B(B, state)
            remember_B(B, state)
    if not bpool:
        raise ValueError("no full-rank Sidon-tail B candidate")

    bfinal = []
    for _, number, B, rows, cols, short, four in bpool:
        spectrum = _component_score(short, cols, rows, four)
        key = spectrum[:4] + (four, spectrum[6], spectrum[5])
        bfinal.append((key, number, B, rows, spectrum))
    bfinal.sort(key=lambda item: item[:2])
    bfinal = bfinal[:3]

    options = []
    for _, ap, A, ar, ac in apool:
        for _, bp, B, br, bc in bfinal:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 598 or k != 260:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            # Small-word protection precedes an uncalibrated decoder
            # probe; optimistic larger distance estimates do not.
            protection = (
                max(ac[0], bc[0]), ac[0]+bc[0],
                max(ac[1], bc[1]), ac[1]+bc[1],
                max(rx[0], rz[0]), rx[0]+rz[0],
                max(rx[1], rz[1]), rx[1]+rz[1],
            )
            geometry_key = (
                max(7*ac[2], 3*bc[2]), 7*ac[2]+3*bc[2],
                max(7*ac[3], 3*bc[3]), 7*ac[3]+3*bc[3],
                ac[4]+bc[4], ac[5]+bc[5], ap, bp,
            )
            options.append((protection, geometry_key, len(options),
                            A, B, hx, hz, ar, br, ac, bc))
    options.sort(key=lambda item: item[:3])
    if not options:
        raise ValueError("no verified k=260 gate-lean CSS finalist")

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two integrated intervals, NOT an ASC memory experiment.

        Decode H*e1+m=s1 and H*(e1+e2)=s2 jointly. The closing
        syndrome is perfect. Gate-degree marginals and randomly
        ordered opposite-check suffix faults are approximations.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length()-1)
                row ^= bit
            return out

        nr = len(checks)
        nv = 2*n+nr
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        ns = [sites(row) for row in augmented]
        width = max(map(len, ns))
        incidence = np.full((2*nr, width), nv, dtype=np.int32)
        for r, row in enumerate(ns):
            incidence[r, :len(row)] = row
        flat = incidence.ravel()
        valid = flat != nv
        ne = len(flat)
        degree = np.bincount(flat[valid], minlength=nv)
        assert np.all(degree > 0)
        ve = np.full((nv+1, int(degree.max())), ne, dtype=np.int32)
        used = np.zeros(nv, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            ve[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([c.bit_count() for c in _columns(checks, n)])
        opposite = np.asarray([
            c.bit_count() for c in _columns(stabilizers, n)])
        levels = np.asarray((0.55, 0.80, 1.05))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+opposite)*2/3
        measure_mu = np.asarray([
            3*0.002*(2+row.bit_count())*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        errors = np.zeros((shots, nv+1), dtype=bool)
        errors[:, :n] = channel.random((shots, n)) < data_p
        errors[:, n:2*n] = channel.random((shots, n)) < data_p
        errors[:, 2*n:nv] = channel.random((shots, nr)) < measure_p

        for row in stabilizers:
            order = channel.permutation(sites(row))
            event_p = np.minimum(
                levels*3*0.002*len(order)*2/3, 0.20)
            for interval in range(2):
                active = np.flatnonzero(channel.random(shots) < event_p)
                for shot in active:
                    cut = int(channel.integers(1, len(order)))
                    errors[shot, interval*n+order[cut:]] ^= True

        marginal_mu = data_mu + 3*0.002*opposite*0.30
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal_p, marginal_p,
             np.clip(measure_p, 0.002, 0.25)), axis=1)
        prior = np.full((shots, nv+1), 30.0)
        prior[:, :nv] = np.log((1-probabilities)/probabilities)
        target = (errors[:, flat].reshape(
            shots, 2*nr, width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        actual_final = errors[:, :n] ^ errors[:, n:2*n]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        wrong = late = work = 0
        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, 2*nr, width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, ne)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, ne+1))
            padded[:, :ne] = outgoing
            posterior = prior + padded[:, ve].sum(axis=2)
            posterior[:, nv] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, 2*nr, width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(
                        decoded_final[local] ^ actual_final[local]):
                    word |= 1 << int(c)
                wrong += int(nontrivial(word))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            actual_final = actual_final[keep]
        unresolved = len(prior)
        return wrong+unresolved, unresolved, late+unresolved, work+37*unresolved

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    seed = 260598713 + 1009*idx
    training = []
    for entry in options[:4]:
        protection, geometry_key, number, A, B, hx, hz, ar, br, ac, bc = entry
        x = noisy_probe(hz, hx, 598, seed, 40)
        z = noisy_probe(hx, hz, 598, seed+1, 40)
        key = protection + quality(x, z) + geometry_key + (number,)
        training.append((key, number, entry, x, z))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, entry, tx, tz in training[:2]:
        protection, geometry_key, _, A, B, hx, hz, ar, br, ac, bc = entry
        x = noisy_probe(hz, hx, 598, seed+1000003, 80)
        z = noisy_probe(hx, hz, 598, seed+1000004, 80)
        # Independent holdout leads. Training and word multiplicities
        # break ties; neither is claimed to be measured circuit pL.
        key = (protection + quality(x, z) + quality(tx, tz)
               + geometry_key + (number,))
        finalists.append((key, entry, tx, tz, x, z))
    finalists.sort(key=lambda item: item[0])
    key, entry, tx, tz, x, z = finalists[0]
    _, _, _, A, B, hx, hz, ar, br, ac, bc = entry

    # Assert that composed probe components equal the actual final lift.
    actual_a = [
        sum(((row >> (j*7*q)) & mask) << (j*q) for j in range(3))
        for row in hx[:q]
    ]
    actual_b = [row & ((1 << (7*q))-1) for row in hz[:2*q]]
    assert actual_a == ar and actual_b == br
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 260
    assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    assert tuple(map(len, A[0])) == (2, 3, 3)
    assert 0.425 <= 260/598 < 0.475
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "design": design, "batch": batch,
            "steps": steps, "k": 260, "cnots": expected_cnots,
            "component_profiles": (ac, bc),
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z), "selection_score": key}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete Sidon-tail, holdout-selected cell without redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree A trinomials, strengthened B columns and spectrum/ambiguity screening suffice for rare collective circuit failures: fitness was 3.9894, slot 6 failed 110/1250 Z and 121/1250 X shots, and all five measured codes retained abundant sector-one weight-six witnesses.
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
