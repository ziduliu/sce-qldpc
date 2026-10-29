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
CONCEPT = ("AGL(1,5), order 20, A=(1,3), B=(2,7): selecting constituents against verified dependencies through six should supply five fresh d30s>=7 readings and improve the cell-0.450 fitness above 16.4; measured searches can refute this.")
GROUP = ("L2 order repair within the same finite-field affine family F_t semidirect F_t^*. The new factor 23 requires q<=26; AGL(1,5), with additive F5 and its full order-four multiplicative complement, has order 20 and is the largest full finite-field affine group fitting that bound. The parent's order-56 affine bed measured five distance-ten leaders, motivating verified constituent screening without assuming those distances transfer.")
SHAPES = ("Target cell 0.450 with (1,3)x(2,7): n=460, kb=200 and guaranteed rate 10/23=0.434783. Constituents have variable/check counts 60/20 and 140/40, rather than equal lengths. Parent measured minima occupied only sector-one Acol/Bcol blocks, prominently Acol1 and Bcol1. Four cubic B columns and three degree-two columns preserve useful density under both caps. Gate complete measured ranks inside 0.425<=k/n<0.475.")
FILLING = ("Twenty cards cross five noncommuting A quotient geometries with four B support/gap policies. A has three trinomials. Every B row has nine terms; its first four columns have degree three and its last three degree two. Hence HX weights are 11..12 and HZ weight twelve. Compile actions exclusively from scaffold unit lifts, exclude constituent kernels through four, screen B through five, and exhaust constituent dependencies through six on retained candidates. Repair witness-heavy columns, retain actual words during equal independent rescreens, and verify complete logical embeddings and stabilizer classes in both sectors. Unequal uncoloured Tanner invariants separate emitted designs. Found readings are upper bounds; bounded misses and constituent exclusions do not certify quantum distance.")
FAMILY_NAME = "r045_affine_F5_C4_13x27_mixeddegree_exact6_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same finite-field affine rule: t -> alpha^b*t+a.
    # F5 is the prime field; alpha=2 generates F5^*.
    if not valid_q(q):
        raise ValueError("AGL(1,5) has order 20")
    p, r, alpha = 5, 4, 2
    els = [(a, b) for b in range(r) for a in range(p)]
    powers = tuple(pow(alpha, b, p) for b in range(r))

    def red(g):
        return (g[0] % p, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % p, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % p, bi)

    def idx(g):
        a, b = red(g)
        return p*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Five different term architectures, crossed with four pair-order recipes.
# For single-row sides: wx=sum(A)+max(B), wz=sum(B)+max(A).
# Their respective (wx,wz) maxima are (9,12), (10,12), (10,12),
# (11,12), and (8,12). No profile contains an empty or singleton cell.
_SUPPORT_PLANS = (
    ((3, 3), (3, 3, 3)),
    ((3, 3), (2, 3, 4)),
    ((3, 4), (2, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((2, 3), (3, 3, 3)),
)


def _designed_grid(rng, G, counts, mode, phase):
    """Constrain internal relative orders, not just translate one labeling."""
    from math import gcd
    divisor = (1, 2, 3, 6)[mode]
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    row = []
    for j, weight in enumerate(counts):
        b = (phase + j) & 1
        # A binomial with an order-four relative element would itself have
        # a weight-four kernel word. Use order 42 for every binomial.
        wanted = 1 if weight == 2 else divisor
        steps = [d for d in range(1, 42) if gcd(d, 42) == wanted]
        first = rng.choice(pools[b])
        second = ((first[0] + rng.choice(steps)) % 42, b)
        cell = [first, second]
        if weight == 3:
            remaining = [1 - b]
        elif weight == 4:
            remaining = [1 - b, 1 - b if mode & 1 else b]
        else:
            remaining = []
        for coset in remaining:
            legal = [
                g for g in pools[coset]
                if g not in cell
                and all(G.mul(G.inv(h), g) != (21, 0) for h in cell)
            ]
            cell.append(rng.choice(legal))
        assert len(cell) == weight and len(set(cell)) == weight
        row.append(cell)
    return [row]


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _component_data(hx, hz, q):
    """Extract both actions from the scaffold's verified LP construction."""
    mask = (1 << q) - 1
    # HX beta=0: the A blocks are sector-one blocks 0 and 3.
    ar = [
        (row & mask) | (((row >> (3*q)) & mask) << q)
        for row in hx[:q]
    ]
    # HZ A-column=0: B occupies the first three sector-one blocks.
    bm = (1 << (3*q)) - 1
    br = [row & bm for row in hz[:q]]
    return ar, br, _columns(ar, 2*q), _columns(br, 3*q)


from functools import lru_cache


@lru_cache(maxsize=256)
def _dic21_norm_lifts(columns, G, max_tests=192):
    """Find actual component witnesses using the central C2 extension.

    All parity data come from verified columns. Group operations only
    partition coordinates. Singleton/pair collisions exhaust quotient
    supports through four. Triple collisions supply weight-six projections;
    at most max_tests distinct projections receive the orientation screen.

    For each tested six-support, all orientations and norm corrections
    involving <=2 orbit pairs are tested. This covers every physical
    weight<=10 completion of THAT projection, not every component word.
    The width+1 sentinel means no witness found.
    """
    q = G.q
    assert q == 84 and len(columns) % q == 0
    assert all(s.bit_length() <= q for s in columns)
    z = (21, 0)
    assert G.mul(z, z) == G.e

    covered, orbits = set(), []
    for g in G.elements:
        a = G.idx(g)
        if a in covered:
            continue
        b = G.idx(G.mul(z, g))
        orbit = tuple(sorted((a, b)))
        assert a != b
        covered.update(orbit)
        orbits.append(orbit)
    assert len(orbits) == 42 and len(covered) == q

    output_masks = [(1 << a) | (1 << b) for a, b in orbits]

    def compact(syndrome):
        out = 0
        for r, (a, b) in enumerate(orbits):
            if syndrome & (1 << a):
                out |= 1 << r
        return out

    folded, representatives, norms, representative_syndromes = [], [], [], []
    for block in range(len(columns) // q):
        for a, b in orbits:
            a, b = block*q+a, block*q+b
            syndrome = columns[a] ^ columns[b]
            assert all((syndrome & mask) in (0, mask)
                       for mask in output_masks)
            folded.append(compact(syndrome))
            representatives.append(1 << a)
            norms.append((1 << a) | (1 << b))
            representative_syndromes.append(compact(columns[a]))

    def expand_norm(support):
        word = 0
        while support:
            bit = support & -support
            word ^= norms[bit.bit_length()-1]
            support ^= bit
        return word

    def verified_weight(word):
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= columns[bit.bit_length()-1]
            remaining ^= bit
        assert word and syndrome == 0
        return word.bit_count()

    # One representative suffices for finding a correction involving
    # at most two norm pairs. Any such correction adds at most four bits.
    pairs = {0: 0}
    bits = [1 << i for i in range(len(folded))]
    for i, syndrome in enumerate(folded):
        old = pairs.get(syndrome)
        if old is not None:
            return verified_weight(expand_norm(bits[i] ^ old))
        pairs[syndrome] = bits[i]
    for i, syndrome in enumerate(folded):
        for j in range(i):
            s = syndrome ^ folded[j]
            support = bits[i] | bits[j]
            old = pairs.get(s)
            if old is not None:
                return verified_weight(expand_norm(support ^ old))
            pairs[s] = support

    best = len(columns) + 1
    triples, tested = {}, set()
    for i in range(2, len(folded)):
        for j in range(1, i):
            partial = folded[i] ^ folded[j]
            for h in range(j):
                syndrome = partial ^ folded[h]
                support = bits[i] | bits[j] | bits[h]
                lower = pairs.get(syndrome)
                if lower is not None:
                    return verified_weight(expand_norm(support ^ lower))
                old = triples.get(syndrome)
                if old is None:
                    triples[syndrome] = support
                    continue
                projection = support ^ old
                if projection in tested:
                    continue
                tested.add(projection)
                assert projection.bit_count() == 6
                # The orbit-constant lift itself is a genuine weight-12
                # witness, even if no lighter orientation is discovered.
                best = min(best, verified_weight(expand_norm(projection)))

                selected = []
                remaining = projection
                while remaining:
                    bit = remaining & -remaining
                    selected.append(bit.bit_length()-1)
                    remaining ^= bit
                word, residual = 0, 0
                for c in selected:
                    word ^= representatives[c]
                    residual ^= representative_syndromes[c]

                # Gray-code enumeration changes one orientation each step.
                for orientation in range(64):
                    if orientation:
                        position = (orientation & -orientation).bit_length()-1
                        c = selected[position]
                        word ^= norms[c]
                        residual ^= folded[c]
                    correction = pairs.get(residual)
                    if correction is not None:
                        trial = word ^ expand_norm(correction)
                        weight = verified_weight(trial)
                        assert weight <= 10
                        return min(best, weight)
                if len(tested) >= max_tests:
                    return best
    return best


def _dic21_connected(columns):
    """Connectivity of the constituent Tanner graph from verified columns."""
    reached_checks = columns[0]
    reached_variables = 0
    while True:
        old = reached_variables
        for i, syndrome in enumerate(columns):
            if syndrome & reached_checks:
                reached_variables |= 1 << i
                reached_checks |= syndrome
        if reached_variables == old:
            return reached_variables.bit_count() == len(columns)


def _dic21_beam(columns, rng, width=192, limit=12):
    """Find actual short constituent dependencies; absence proves nothing.

    Each occupied protograph block supplies a regular-action anchor.
    Odd-check completion and equal-boundary collisions generate only
    genuine zero-syndrome words. Beam truncation makes this heuristic.
    """
    height = max(s.bit_length() for s in columns)
    incident = [[] for _ in range(height)]
    for i, syndrome in enumerate(columns):
        while syndrome:
            bit = syndrome & -syndrome
            incident[bit.bit_length() - 1].append(i)
            syndrome ^= bit
    bits = [1 << i for i in range(len(columns))]
    best = len(columns) + 1

    for anchor in range(0, len(columns), 84):
        for restart in range(2):
            states = [(bits[anchor], columns[anchor])]
            memory = {0: 0, columns[anchor]: bits[anchor]}
            for depth in range(1, limit):
                following = {}
                for word, syndrome in states:
                    choices = None
                    remaining = syndrome
                    while remaining:
                        bit = remaining & -remaining
                        check = bit.bit_length() - 1
                        available = [i for i in incident[check]
                                     if not word & bits[i]]
                        if choices is None or len(available) < len(choices):
                            choices = available
                        remaining ^= bit
                    for i in choices or ():
                        new_word = word | bits[i]
                        new_syndrome = syndrome ^ columns[i]
                        old = memory.get(new_syndrome)
                        if old is not None:
                            dependency = new_word ^ old
                            if dependency:
                                best = min(best, dependency.bit_count())
                        else:
                            memory[new_syndrome] = new_word
                        if new_syndrome == 0:
                            best = min(best, new_word.bit_count())
                            continue
                        if best <= 10:
                            return best
                        weight = depth + 1
                        if (weight < min(best, limit + 1)
                                and new_syndrome.bit_count()
                                <= 3 * (limit - weight)):
                            following[new_word] = new_syndrome
                if not following:
                    break

                # Preserve several boundary-weight classes instead of
                # retaining only one attractive trapping-set geometry.
                buckets = {}
                for word, syndrome in following.items():
                    buckets.setdefault(syndrome.bit_count(), []).append(
                        (word, syndrome))
                quota = max(1, width // len(buckets))
                states, spare = [], []
                for boundary in sorted(buckets):
                    bucket = buckets[boundary]
                    rng.shuffle(bucket)
                    states.extend(bucket[:quota])
                    spare.extend(bucket[quota:])
                if len(states) < width:
                    rng.shuffle(spare)
                    states.extend(spare[:width - len(states)])
                states = states[:width]
    return best


def _dic21_fingerprint(hx, hz, n, k):
    """Permutation invariant of the combined, uncoloured Tanner graph.

    Unequal fingerprints certify nonisomorphism. Equal fingerprints are
    conservatively rejected, not asserted to prove isomorphism.
    """
    from collections import Counter
    rows = hx + hz
    size = len(rows)
    adjacency = [0] * size
    overlaps = [[0] * size for _ in rows]
    weights = [row.bit_count() for row in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i][j] = overlaps[j][i] = overlap
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = []
    for i in range(size):
        histogram = Counter(
            (weights[j], overlaps[i][j],
             (adjacency[i] & adjacency[j]).bit_count())
            for j in range(size) if j != i)
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    return n, k, tuple(sorted(Counter(profiles).items()))


def _optimise_dic21(rng, G, idx, used):
    """Optimize one prescribed architecture, excluding bank equivalents.

    Counts, actions, ranks and stabilizer spaces use the verified lift.
    The new deterministic stream does not revisit the parental draws.
    """
    from math import gcd
    rng = random.Random(rng.getrandbits(64) ^ 0xD1C21140704)
    card = idx % 20
    a_cards = ((3, 3), (3, 6), (6, 6), (1, 3), (2, 6))
    b_cards = ((3, 3, 3), (3, 3, 6), (3, 6, 6), (6, 6, 6))
    ad = a_cards[card // 4]
    bd = b_cards[card % 4]
    # gcd(d,42)=1,2,3,6 gives rotation-pair orders 42,21,14,7.
    steps = {
        divisor: [d for d in range(1, 42) if gcd(d, 42) == divisor]
        for divisor in (1, 2, 3, 6)
    }
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def packed(divisors):
        """Pack ordered check-pair orbits, not copies of a labeled code.

        The relative ratios are only a construction filter. The actual
        component four-cycle count is checked on the verified matrices.
        """
        for restart in range(24):
            used = set()
            row = [None] * len(divisors)
            order = list(range(len(divisors)))
            rng.shuffle(order)
            for j in order:
                for attempt in range(192):
                    a, b = rng.randrange(42), rng.randrange(2)
                    d = rng.choice(steps[divisors[j]])
                    cell = [(a, b), ((a + d) % 42, b),
                            (rng.randrange(42), 1 - b)]
                    differences = {
                        G._key(G.mul(g, G.inv(h)))
                        for g in cell for h in cell if g != h
                    }
                    if len(differences) != 6:
                        continue
                    if (21, 0) in differences or differences & used:
                        continue
                    used.update(differences)
                    row[j] = cell
                    break
                if row[j] is None:
                    break
            if all(cell is not None for cell in row):
                return [row]
        return None

    def six_cycles(cols):
        """Exact Tanner six-cycle count for a four-cycle-free cubic matrix.

        Each cubic variable contributes one same-variable triangle in the
        check adjacency graph; subtract those from its triangle count.
        """
        height = max((s.bit_length() for s in cols), default=0)
        adjacency = [0] * height
        for syndrome in cols:
            neighbors = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbors.append(bit.bit_length() - 1)
                syndrome ^= bit
            assert len(neighbors) == 3
            for a in neighbors:
                for b in neighbors:
                    if a != b:
                        adjacency[a] |= 1 << b
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            higher = neighbors >> (a + 1)
            while higher:
                bit = higher & -higher
                b = a + 1 + bit.bit_length() - 1
                triangles += (
                    (neighbors & adjacency[b]) >> (b + 1)
                ).bit_count()
                higher ^= bit
        return triangles - len(cols)

    def row_basis(rows):
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

    def outside(word, basis):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def probe(cols, trials, start=None, stabilizers=None, generators=()):
        """Actual kernel/logical witnesses with whole-basis reduction.

        Every XOR remains in the verified kernel. For quantum searches,
        only words outside the matching stabilizer span lower the reading.
        A no-witness sentinel is never interpreted as a certificate.
        """
        best = len(cols) + 1 if start is None else start
        order = list(range(len(cols)))
        memory = []

        def consider(word):
            nonlocal best
            if (word and word.bit_count() < best
                    and (stabilizers is None or outside(word, stabilizers))):
                best = word.bit_count()

        for trial in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    words.append(word)
                    consider(word)

            # Reduce every basis direction, not just six short seeds.
            # Memory directions can help without needing to preserve an
            # independent basis: only genuine nonzero witnesses are scored.
            sweeps = 2 if trials > 8 and stabilizers is None else 1
            for sweep in range(sweeps):
                rng.shuffle(words)
                changed = False
                for i in range(len(words)):
                    word = words[i]
                    weight = word.bit_count()
                    for j, other in enumerate(words):
                        if i == j:
                            continue
                        combined = word ^ other
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                            changed = True
                    for other in memory:
                        combined = word ^ other
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                            changed = True
                    words[i] = word
                    consider(word)
                if not changed:
                    break

            # Raw check rows, unlike echelon rows, are sparse dressing
            # moves. Single/pair additions preserve the logical class.
            dressed = []
            if generators:
                seeds = sorted(set(words + memory), key=int.bit_count)
                for seed in seeds[:12]:
                    if not outside(seed, stabilizers):
                        continue
                    word = seed
                    for sweep in range(2):
                        chosen, weight = word, word.bit_count()
                        nearby = sorted(
                            (((word ^ row).bit_count(), row)
                             for row in generators),
                            key=lambda item: item[0])[:12]
                        for _, row in nearby:
                            trial = word ^ row
                            if trial.bit_count() < weight:
                                chosen, weight = trial, trial.bit_count()
                        for a, (_, row) in enumerate(nearby):
                            for _, other in nearby[:a]:
                                trial = word ^ row ^ other
                                if trial.bit_count() < weight:
                                    chosen, weight = trial, trial.bit_count()
                        if chosen == word:
                            break
                        word = chosen
                        consider(word)
                    dressed.append(word)

            discoveries = set(words)
            discoveries.update(memory)
            discoveries.update(dressed)
            short = sorted(discoveries, key=int.bit_count)[:48]
            cutoff = min(best + 4, len(cols))
            for i, word in enumerate(short):
                for other in short[:i]:
                    combined = word ^ other
                    if combined and combined.bit_count() <= cutoff:
                        consider(combined)
                        discoveries.add(combined)
            small = sorted(discoveries, key=int.bit_count)[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        combined = pair ^ small[h]
                        if combined and combined.bit_count() <= cutoff:
                            consider(combined)
                            discoveries.add(combined)
            memory = sorted(discoveries, key=int.bit_count)[:48]

            # Once a constituent already witnesses the old bottleneck,
            # spend the remaining budget on other candidate fillings.
            if ((stabilizers is None and best <= 10)
                    or (stabilizers is not None and best <= 8)):
                break
        return best

    def score(candidate):
        d = min(candidate["da"], candidate["db"], candidate["full"])
        return (candidate["k"] * d * d / candidate["n"], d,
                candidate["da"] + candidate["db"],
                -candidate["six"][1], -sum(candidate["six"]))

    # Optimize the measured X bottleneck independently. Internal stubs
    # only extract verified actions; final slots still use five trinomials.
    stub_a = [[[G.e], [G.e]]]
    b_pool = []

    def b_key(item):
        return (min(item["reading"], 14), -item["six"], -item["attempt"])

    for attempt in range(96):
        B = packed(bd)
        if B is None:
            continue
        hx, hz, _ = lifted_matrices(G, stub_a, B)
        _, br, _, bc = _component_data(hx, hz, G.q)
        if _component_cycles(br) or not _dic21_connected(bc):
            continue
        norm = _dic21_norm_lifts(tuple(bc), G, 12)
        if norm <= 4:
            continue
        reading = min(norm, probe(bc, 2))
        if reading > len(bc):
            continue
        item = {
            "B": B, "bc": bc, "norm": norm,
            "reading": reading, "six": six_cycles(bc),
            "attempt": attempt,
        }
        b_pool.append(item)
        b_pool.sort(key=b_key, reverse=True)
        del b_pool[12:]

    # Equal independent rescreening budgets avoid letting one lucky
    # short information set determine the chosen B constituent.
    for item in b_pool:
        item["norm"] = min(
            item["norm"], _dic21_norm_lifts(tuple(item["bc"]), G, 192))
        item["reading"] = min(
            item["reading"], item["norm"],
            probe(item["bc"], 12, item["reading"]))
    b_pool.sort(key=b_key, reverse=True)

    raw = []
    for item in b_pool[:4]:
        for attempt in range(8):
            A, B = packed(ad), item["B"]
            if A is None:
                continue
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br, ac, bc = _component_data(hx, hz, G.q)
            if _component_cycles(ar) or _component_cycles(br):
                continue
            if not (_dic21_connected(ac) and _dic21_connected(bc)):
                continue
            norm_a = _dic21_norm_lifts(tuple(ac), G, 32)
            if norm_a <= 4:
                continue
            ta, tb = six_cycles(ac), item["six"]
            structure = (
                -min(norm_a, item["reading"], 14),
                2*tb+ta, max(ta, tb), tb, ta)
            candidate = {
                "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
                "ac": ac, "bc": bc, "six": (ta, tb),
                "norm_a": norm_a, "norm_b": item["norm"],
                "b_reading": item["reading"],
                "structure": structure, "full": n + 1,
            }
            raw.append(candidate)
            raw.sort(key=lambda c: c["structure"])
            del raw[16:]

    pool = []
    for candidate in raw:
        hx, hz, n = candidate["hx"], candidate["hz"], candidate["n"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 168:
            continue
        candidate["k"] = k
        if min(_short_floor(candidate["ac"]),
               _short_floor(candidate["bc"])) < 5:
            continue
        candidate["da"] = min(
            candidate["norm_a"], probe(candidate["ac"], 4))
        candidate["db"] = min(
            candidate["norm_b"], candidate["b_reading"],
            probe(candidate["bc"], 8))
        # Require actual component witnesses, rather than scoring a
        # no-witness sentinel as a large distance.
        if (candidate["da"] > len(candidate["ac"])
                or candidate["db"] > len(candidate["bc"])):
            continue
        pool.append(candidate)

    if not pool:
        raise ValueError("no rank-legal packed cubic candidate")

    finalists = sorted(pool, key=score, reverse=True)[:4]
    for candidate in finalists:
        candidate["da"] = probe(
            candidate["ac"], 16, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 40, candidate["db"])
        if candidate["db"] > 10:
            candidate["db"] = min(
                candidate["db"],
                _dic21_beam(candidate["bc"], rng, 192, 12))
        candidate["xs"] = row_basis(candidate["hx"])
        candidate["zs"] = row_basis(candidate["hz"])
        candidate["xc"] = _columns(candidate["hz"], candidate["n"])
        candidate["zc"] = _columns(candidate["hx"], candidate["n"])
        # Start without an assumed constituent-to-logical embedding.
        # These two readings lower only on verified logical witnesses.
        candidate["dx"] = probe(
            candidate["xc"], 16, None, candidate["xs"],
            candidate["hx"])
        candidate["dz"] = probe(
            candidate["zc"], 16, None, candidate["zs"],
            candidate["hz"])
        candidate["full"] = min(candidate["dx"], candidate["dz"])

    leaders = sorted(finalists, key=score, reverse=True)[:2]
    for candidate in leaders:
        candidate["da"] = min(
            candidate["da"],
            _dic21_norm_lifts(tuple(candidate["ac"]), G, 192))
        candidate["db"] = min(
            candidate["db"],
            _dic21_norm_lifts(tuple(candidate["bc"]), G, 384))
        candidate["da"] = probe(
            candidate["ac"], 24, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 80, candidate["db"])
        if candidate["db"] > 10:
            candidate["db"] = min(
                candidate["db"],
                _dic21_beam(candidate["bc"], rng, 320, 12))
        candidate["dx"] = probe(
            candidate["xc"], 24, candidate["dx"], candidate["xs"],
            candidate["hx"])
        candidate["dz"] = probe(
            candidate["zc"], 24, candidate["dz"], candidate["zs"],
            candidate["hz"])
        candidate["full"] = min(candidate["dx"], candidate["dz"])

    for winner in sorted(finalists, key=score, reverse=True):
        fingerprint = _dic21_fingerprint(
            winner["hx"], winner["hz"], winner["n"], winner["k"])
        if fingerprint in used:
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["k"] == 168
        assert max(row.bit_count() for row in
                   winner["hx"] + winner["hz"]) <= W_MAX
        assert all(len(cell) == 3 and len(set(cell)) == 3
                   for M in (winner["A"], winner["B"])
                   for row in M for cell in row)
        return {
            "idx": idx, "card": card,
            "cells": {"A": winner["A"], "B": winner["B"]},
            "screen": score(winner), "six": winner["six"],
            "fingerprint": fingerprint,
        }
    return None


from functools import lru_cache


@lru_cache(maxsize=1)
def _dic21_bank(G):
    """Deterministically couple slots only for invariant rejection.

    Each card is independently optimized. No selected labeling is
    translated, permuted or automorphism-copied into another slot.
    """
    used, bank = set(), []
    for idx in range(N_SLOTS):
        local = random.Random(BASE_SEED * 100003 + 104729 * idx + 140704)
        chosen = None
        for batch in range(3):
            try:
                chosen = _optimise_dic21(local, G, idx, used)
            except ValueError:
                chosen = None
            if chosen is not None:
                break
        if chosen is not None:
            used.add(chosen["fingerprint"])
        bank.append(chosen)
    return tuple(bank)


# Active affine-group optimizer. Historical Dic21 routines above are inactive.
_AG_POWERS = (1, 2, 4, 3, 6, 7, 5)
_AG_ARCHITECTURES = ((0, 0), (0, 1), (0, 2), (1, 1), (2, 3))


def _ag_field_mul(a, b):
    """Field arithmetic constrains labels; it never constructs a lift."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 8:
            a ^= 11
    return result


def _ag_basis(rows):
    basis = {}
    for original in rows:
        word = original
        while word:
            p = word.bit_length()-1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _ag_outside(word, basis):
    while word:
        p = word.bit_length()-1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _ag_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _ag_search(cols, rng, trials, seeds=(), stabilizers=None):
    """Retain real kernel/logical words, never a claimed exclusion bound.

    Short kernel directions and short logical representatives have
    separate memories, so sparse stabilizers cannot crowd out logicals.
    """
    width = len(cols)
    span = None if stabilizers is None else _ag_basis(stabilizers)
    generators = [] if stabilizers is None else list(stabilizers)
    best, found = width+1, set()
    memory = []

    def observe(word):
        nonlocal best, found
        if not word or (span is not None and not _ag_outside(word, span)):
            return
        weight = word.bit_count()
        best = min(best, weight)
        found.add(word)

    for word in seeds:
        assert _ag_syndrome(cols, word) == 0
        observe(word)
        if word:
            memory.append(word)
    memory = sorted(set(memory), key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(width))

    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)

        directions = sorted(set(dependencies+memory),
                            key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        # Whole-basis descent using a bounded set of short directions.
        for sweep in range(2):
            reducers = directions[:32]
            reduced = []
            for word in directions:
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                observe(word)
                reduced.append(word)
            directions = sorted(set(reduced),
                                key=lambda w: (w.bit_count(), w))
            discoveries.update(directions[:40])

        short = directions[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        if generators:
            logicals = sorted(found, key=lambda w: (w.bit_count(), w))[:8]
            for word in logicals:
                # Kernel additions may change the logical class, so
                # explicitly retain a nonzero class during this descent.
                for sweep in range(2):
                    changed = False
                    for direction in dependencies+memory:
                        candidate = word ^ direction
                        if (candidate.bit_count() < word.bit_count()
                                and _ag_outside(candidate, span)):
                            word = candidate
                            changed = True
                    if not changed:
                        break
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in generators),
                        key=lambda x: x[0])[:14]
                    for _, row in nearby:
                        candidate = word ^ row
                        if candidate.bit_count() < weight:
                            chosen, weight = candidate, candidate.bit_count()
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            if candidate.bit_count() < weight:
                                chosen, weight = candidate, candidate.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                assert _ag_outside(word, span)
                observe(word)
                discoveries.add(word)

        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        found = set(sorted(found,
                           key=lambda w: (w.bit_count(), w))[:24])
        if best <= 4:
            break
    words = tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:24])
    assert all(_ag_syndrome(cols, w) == 0 for w in words)
    return best, words


def _ag_beam(cols, q, rng, width=144, limit=12):
    """Bounded anchored constituent dependency search; misses prove nothing."""
    height = max(s.bit_length() for s in cols)
    incident = [[] for _ in range(height)]
    bits = [1 << i for i in range(len(cols))]
    for c, original in enumerate(cols):
        syndrome = original
        while syndrome:
            bit = syndrome & -syndrome
            incident[bit.bit_length()-1].append(c)
            syndrome ^= bit
    found = set()

    for anchor in range(0, len(cols), q):
        states = [(bits[anchor], cols[anchor])]
        memory = {0: 0, cols[anchor]: bits[anchor]}
        for depth in range(1, limit):
            following = {}
            for word, syndrome in states:
                choices = None
                remaining = syndrome
                while remaining:
                    bit = remaining & -remaining
                    available = [c for c in incident[bit.bit_length()-1]
                                 if not word & bits[c]]
                    if choices is None or len(available) < len(choices):
                        choices = available
                    remaining ^= bit
                for c in choices or ():
                    nw, ns = word | bits[c], syndrome ^ cols[c]
                    previous = memory.get(ns)
                    if previous is not None:
                        dependency = nw ^ previous
                        if dependency:
                            found.add(dependency)
                    else:
                        memory[ns] = nw
                    if ns == 0:
                        found.add(nw)
                    elif ns.bit_count() <= 3*(limit-depth-1):
                        following[nw] = ns
            if not following:
                break
            buckets = {}
            for word, syndrome in following.items():
                buckets.setdefault(syndrome.bit_count(), []).append(
                    (word, syndrome))
            quota = max(1, width//len(buckets))
            states, spare = [], []
            for boundary in sorted(buckets):
                bucket = buckets[boundary]
                rng.shuffle(bucket)
                states.extend(bucket[:quota])
                spare.extend(bucket[quota:])
            rng.shuffle(spare)
            states.extend(spare[:max(0, width-len(states))])
            states = states[:width]
    words = tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:24])
    assert all(_ag_syndrome(cols, w) == 0 for w in words)
    return words


def _ag_draw(rng, G, non_ap, mode):
    """Fresh quotient geometries and normal coordinates, not orbit copies."""
    patterns = [(0, 1, 3)]*non_ap + [(0, 1, 2)]*(3-non_ap)
    rng.shuffle(patterns)
    row, used = [], set()
    for pattern in patterns:
        chosen = None
        for attempt in range(64):
            phase, unit = rng.randrange(7), rng.randrange(1, 7)
            bs = [(phase+unit*b) % 7 for b in pattern]
            if mode == 0:
                aa = rng.sample(range(8), 3)
            elif mode == 1:
                base, vector = rng.randrange(8), rng.randrange(1, 8)
                aa = [base ^ _ag_field_mul(vector, t) for t in (0, 1, 2)]
            elif mode == 2:
                base = rng.randrange(8)
                aa = [base, base, base ^ rng.randrange(1, 8)]
                rng.shuffle(aa)
            else:
                u, v = rng.sample(range(1, 8), 2)
                aa = [u, v, u ^ v]
                rng.shuffle(aa)
            cell = list(zip(aa, bs))
            anchor = G.inv(cell[0])
            relatives = [G.mul(anchor, g) for g in cell[1:]]
            if G.mul(*relatives) == G.mul(*reversed(relatives)):
                continue
            differences = [
                G.mul(g, G.inv(h))
                for g in cell for h in cell if g != h]
            penalty = (len(differences)-len(set(differences))
                       + 2*len(set(differences) & used))
            key = (penalty, rng.random())
            if chosen is None or key < chosen[0]:
                chosen = (key, cell, set(differences))
            if penalty == 0:
                break
        if chosen is None:
            raise ValueError("affine cubic recipe exhausted")
        _, cell, differences = chosen
        used.update(differences)
        row.append(cell)
    return [row]


def _ag_low4(cols, stabilizers):
    """Exact logical existence through four, using supplied verified columns."""
    span = _ag_basis(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif _ag_outside(word ^ previous, span):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif _ag_outside(word ^ previous, span):
                return True
    return False


def _ag_component_word(cols, q, through):
    """Exact constituent dependency search after exclusion through four.

    Arbitrary column degrees are supported. Each constituent block has
    a transitive commuting regular action, so one anchor per block
    covers its support orbits. This reduction is not applied to the
    complete quantum matrices. There is no search cutoff.
    """
    bits = [1 << i for i in range(len(cols))]
    height = max(c.bit_length() for c in cols)
    degree = max(c.bit_count() for c in cols)
    incidence = [0]*height
    tails = {0: 0}
    for i, original in enumerate(cols):
        assert original not in tails
        tails[original] = bits[i]
        value = original
        while value:
            bit = value & -value
            incidence[bit.bit_length()-1] |= bits[i]
            value ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            key = value ^ cols[j]
            assert key not in tails
            tails[key] = bits[i] | bits[j]

    def visit(value, remaining, banned, word):
        if not value:
            return word
        if remaining <= 0 or value.bit_count() > degree*remaining:
            return 0
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not tail & banned):
                return word | tail
            return 0
        pending, choices, least = value, 0, len(cols)+1
        while pending:
            bit = pending & -pending
            options = incidence[bit.bit_length()-1] & ~banned
            count = options.bit_count()
            if not count:
                return 0
            if count < least:
                choices, least = options, count
            pending ^= bit
        alternatives = []
        while choices:
            bit = choices & -choices
            choices ^= bit
            alternatives.append(bit.bit_length()-1)
        alternatives.sort(
            key=lambda i: (-(cols[i] & value).bit_count(), i))
        excluded = banned
        for i in alternatives:
            excluded |= bits[i]
            answer = visit(value ^ cols[i], remaining-1,
                           excluded, word | bits[i])
            if answer:
                return answer
        return 0

    for anchor in range(0, len(cols), q):
        word = visit(cols[anchor], through-1,
                     bits[anchor], bits[anchor])
        if word:
            assert word.bit_count() <= through
            assert _ag_syndrome(cols, word) == 0
            return word
    return 0


@lru_cache(maxsize=1)
def _ag_bank(G):
    """New unequal-shape affine bank; historical F8 draws are inactive."""
    assert G.q == 20
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    q, mask = G.q, (1 << G.q)-1
    za = [[[], [], []]]
    zb = [[[] for _ in range(7)] for _ in range(2)]
    sa = [[3, 3, 3]]

    a_recipes = (
        ((1, 2), (1, 2), (1, 2)),
        ((1, 2), (1, 3), (2, 3)),
        ((0, 1), (1, 2), (1, 3)),
        ((0, 1), (0, 3), (1, 2)),
        ((0, 2), (1, 2), (1, 3)),
    )
    b_profiles = (
        ((2, 2, 1, 1, 1, 1, 1),
         (1, 1, 2, 2, 1, 1, 1)),
        ((3, 2, 1, 0, 1, 1, 1),
         (0, 1, 2, 3, 1, 1, 1)),
        ((3, 1, 1, 1, 1, 1, 1),
         (0, 2, 2, 2, 1, 1, 1)),
        ((2, 2, 1, 1, 1, 1, 1),
         (1, 1, 2, 2, 1, 1, 1)),
    )
    for sb in b_profiles:
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert tuple(map(sum, sb)) == (9, 9)
        assert [sb[0][j]+sb[1][j] for j in range(7)] \
            == [3, 3, 3, 3, 2, 2, 2]

    # The only parity actions used below come from verified unit lifts.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 460
            source = hx if side == "A" else hz
            block = tuple(_columns(
                [row & mask for row in source[:q]], q))
            assert all(c.bit_count() == 1 for c in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        width = len(grid[0])*q
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, contribution in enumerate(
                            atlas[side][G.idx(g)]):
                        cols[j*q+t] ^= contribution << (i*q)
        rows = [0]*(len(grid)*q)
        for c, original in enumerate(cols):
            value = original
            while value:
                bit = value & -value
                rows[bit.bit_length()-1] |= 1 << c
                value ^= bit
        return rows, tuple(cols)

    def element_order(g):
        h = G.e
        for order in range(1, q+1):
            h = G.mul(h, g)
            if h == G.e:
                return order
        raise ValueError("invalid affine element order")

    long_elements = {
        order: tuple(g for g in G.elements
                     if element_order(g) == order)
        for order in (4, 5)
    }

    def triple(local, pair):
        for attempt in range(96):
            b, d = pair
            g = (local.randrange(5), b)
            h = (local.randrange(5), d)
            cell = (G.e, g, h)
            if len(set(cell)) != 3:
                continue
            if G.mul(g, h) != G.mul(h, g):
                return cell
        return None

    def draw(local, side, policy, lane):
        if side == "A":
            row = []
            for j, pair in enumerate(a_recipes[policy]):
                # Independent relative quotient changes, not a global
                # automorphism image of a completed design.
                twist = local.choice((1, 3)) if lane else 1
                pair = tuple(twist*b % 4 for b in pair)
                cell = triple(local, pair)
                if cell is None:
                    return None
                row.append(cell)
            return (tuple(row),)

        counts = b_profiles[policy]
        grid = [[None]*7 for _ in range(2)]
        relatives = local.sample(G.elements, 3)
        for j in range(4, 7):
            grid[0][j] = (relatives[j-4],)
            grid[1][j] = (G.e,)
        for i in range(2):
            for j in range(4):
                count = counts[i][j]
                if count == 0:
                    cell = ()
                elif count == 1:
                    cell = (local.choice(G.elements),)
                elif count == 2:
                    wanted = 4 if policy == 3 else 5
                    if policy == 1:
                        wanted = (4, 5)[(i+j+lane) & 1]
                    first = local.choice(G.elements)
                    relative = local.choice(long_elements[wanted])
                    cell = (first, G.mul(first, relative))
                else:
                    pair = local.choice(((1, 2), (1, 3), (0, 1)))
                    cell = triple(local, pair)
                    if cell is None:
                        return None
                    gauge = local.choice(G.elements)
                    cell = tuple(G.mul(gauge, g) for g in cell)
                assert len(cell) == len(set(cell)) == count
                grid[i][j] = cell
        return tuple(tuple(row) for row in grid)

    def component_key(item):
        load = sum(w.bit_count() <= item["d"]
                   for w in item["words"])
        return (min(item["d"], 10), item.get("floor", 5),
                -load, -item["rank"], -item["cycles"],
                -item["serial"])

    @lru_cache(maxsize=None)
    def component_pool(side, policy, lane):
        local = random.Random(
            BASE_SEED*100003 + 20450237 + 104729*policy
            + 1000003*int(side == "B") + 15485863*lane)
        pool, seen = [], set()
        serial = 0
        width = 3 if side == "A" else 7

        def admit(grid):
            nonlocal serial
            serial += 1
            if grid is None or grid in seen:
                return
            seen.add(grid)
            rows, cols = compile_component(side, grid)
            expected = ([3]*3 if side == "A"
                        else [3, 3, 3, 3, 2, 2, 2])
            assert all(cols[j*q+t].bit_count() == expected[j]
                       for j in range(width) for t in range(q))
            if _short_floor(cols) < 5:
                return
            rank = gf2_rank(rows)
            if side == "A" and rank < q-4:
                return

            # Cubic A columns exclude odd weights automatically.
            # B mixes degrees two and three, so explicitly test five.
            witness = (0 if side == "A"
                       else _ag_component_word(cols, q, 5))
            if witness:
                return
            d, words = _ag_search(cols, local, 3)
            if not words:
                return
            pool.append(dict(
                grid=grid, cols=cols, rank=rank, d=d, words=words,
                cycles=_component_cycles(rows), serial=serial))
            pool.sort(key=component_key, reverse=True)
            del pool[12:]

        quota = 192 if side == "A" else 320
        for attempt in range(quota):
            admit(draw(local, side, policy, lane))

        # Repair real column voltages, keeping the prescribed count grid.
        # Changed matrices never inherit unverified old witnesses.
        for step in range(48):
            fresh = draw(local, side, policy, lane)
            if fresh is None:
                continue
            if not pool:
                admit(fresh)
                continue
            parent = local.choice(pool[:min(5, len(pool))])
            pressure = [
                1+sum(((w >> (j*q)) & mask).bit_count()
                      for w in parent["words"][:8])
                for j in range(width)]
            j = local.choices(range(width), weights=pressure, k=1)[0]
            grid = [list(row) for row in parent["grid"]]
            for i in range(len(grid)):
                grid[i][j] = fresh[i][j]
            if step % 8 == 7:
                h = (j+1+local.randrange(width-1)) % width
                for i in range(len(grid)):
                    grid[i][h] = fresh[i][h]
            admit(tuple(tuple(row) for row in grid))

        for item in pool:
            witness = _ag_component_word(item["cols"], q, 6)
            item["floor"] = witness.bit_count() if witness else 7
            seeds = item["words"] + ((witness,) if witness else ())
            d, words = _ag_search(item["cols"], local, 20, seeds)
            item["d"], item["words"] = min(item["d"], d), words
            assert item["d"] >= item["floor"]
        pool.sort(key=component_key, reverse=True)
        leaders = pool[:5]
        for item in leaders:
            d, words = _ag_search(
                item["cols"], local, 40, item["words"])
            item["d"], item["words"] = min(item["d"], d), words
        leaders.sort(key=component_key, reverse=True)
        return tuple(leaders)

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("dx", 1000), item.get("dz", 1000), 10)
        return (item["k"]*d*d/item["n"], d,
                item["a"]["floor"]+item["b"]["floor"],
                -item["a"]["cycles"]-item["b"]["cycles"])

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        architecture, policy = divmod(slot % 20, 4)
        lane = slot//20
        ap = component_pool("A", architecture, lane)
        bp = component_pool("B", policy, lane)
        local = random.Random(
            BASE_SEED*100007 + 20450461 + 104729*slot)
        proposals = []
        for a in ap:
            for b in bp:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 460 and 425*n <= 1000*k < 475*n):
                    continue

                # Check both compiled components against complete lifts.
                ar = [
                    sum(((row >> (7*j*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (7*q))-1)
                      for row in hz[:2*q]]
                assert tuple(_columns(ar, 3*q)) == a["cols"]
                assert tuple(_columns(br, 7*q)) == b["cols"]
                proposals.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
        proposals.sort(key=merit, reverse=True)
        finalists = []

        for item in proposals:
            hx, hz, n = item["hx"], item["hz"], item["n"]
            # Omit k from the uncoloured Tanner invariant.
            fp = _dic21_fingerprint(hx, hz, n, 0)
            if fp in emitted:
                continue
            xc, zc = _columns(hz, n), _columns(hx, n)
            if _ag_low4(xc, hx) or _ag_low4(zc, hz):
                continue
            bx = tuple(
                word << (7*j*q)
                for word in item["b"]["words"][:8]
                for j in range(3))
            az = tuple(
                sum(((word >> (j*q)) & mask)
                    << ((7*j+beta)*q) for j in range(3))
                for word in item["a"]["words"][:8]
                for beta in range(7))
            assert all(_ag_syndrome(xc, w) == 0 for w in bx)
            assert all(_ag_syndrome(zc, w) == 0 for w in az)
            dx, wx = _ag_search(xc, local, 3, bx, hx)
            dz, wz = _ag_search(zc, local, 3, az, hz)
            if dx > n or dz > n:
                continue
            item.update(
                xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                bx=bx, az=az, fp=fp)
            finalists.append(item)
            if len(finalists) == 4:
                break

        # Equal independent full-code budgets for every finalist.
        for item in finalists:
            dx, wx = _ag_search(
                item["xc"], local, 8,
                item["bx"]+item["wx"], item["hx"])
            dz, wz = _ag_search(
                item["zc"], local, 8,
                item["az"]+item["wz"], item["hz"])
            item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
            item["wx"], item["wz"] = wx, wz

        if not finalists:
            bank.append(None)
            continue
        winner = max(finalists, key=merit)
        A, B = winner["A"], winner["B"]
        hx, hz = winner["hx"], winner["hz"]
        counts_a = [[len(c) for c in row] for row in A]
        counts_b = [[len(c) for c in row] for row in B]
        assert counts_a == sa
        assert tuple(tuple(row) for row in counts_b) == b_profiles[policy]
        assert _grid_weights(counts_a, counts_b) == 12
        assert _no_dead_lines(counts_a, counts_b)
        assert winner["n"] == 460 and winner["k"] >= 200
        assert 425*460 <= 1000*winner["k"] < 475*460
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= W_MAX
        assert all(len(c) == len(set(c)) <= 3
                   for grid in (A, B) for row in grid for c in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={
                side: tuple(tuple(tuple(c) for c in row)
                            for row in winner[side])
                for side in ("A", "B")},
            k=winner["k"], card=(architecture, policy, lane),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the fresh unequal-shape affine structural card."""
    selected = _ag_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh cell-0.450 affine design passed screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further draws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Order-42 binomials preserve the cubic bed's distance advantage; every distinct binomial-profile parent slot read d2s=5, while all five measured codes used five trinomials.
# REFUTED: Component-kernel selection alone removes the parent's mixed-sector bottleneck; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Cubic pair packing, component four-cycle exclusion and six-cycle minimization suffice to exclude weight-ten sector-one X logicals; current slots 3,7,43 each retained at least 300 such measured operators.
# REFUTED: Connected cubic components, whole-basis reduction and anchored dependency beams suffice for five measured readings >=12; the current parent measured 10,10,12,9,12, including a mixed-sector weight-nine X witness in slot 27.
# REFUTED: Restricting Dic21 B rotation-pair orders to 14/7 with central-quotient and dressed-probe selection eliminates mixed-sector weight-nine witnesses; measured slots 12 and 24 retained such X witnesses, and the five readings were 11,12,9,12,9.
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
