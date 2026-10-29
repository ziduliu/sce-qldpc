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
CONCEPT = ("SL(2,3) x C3, order 72, A=(1,2), B=(1,3): matrix-generating cubic constituents selected against short dependencies and singleton-sector-2 logical witnesses should exceed the cell-0.300 frontier fitness 37.9538. Evaluation can refute this; search readings are not distance certificates.")
GROUP = ("L3 jump from Dic21 to the direct product of determinant-one 2x2 matrices over F3 with C3. The matrix factor has 24 elements, hence q=72 and n=504. The product has centre C2 x C3 and is genuinely different from the order-84 parent. Strong order-117 nonabelian beds motivate testing another noncommutative construction, without asserting that their distance transfers. This exact matrix/direct-product bed is not identified in BEDS.")
SHAPES = ("Keep (1,2)x(1,3) for cell 0.300: n=504, kb=144, guaranteed rate 2/7, and component variables/checks 144/72 and 216/72. The size window permits q<=85. Parent minimum X witnesses of weights nine and eleven touched both Acol/Bcol blocks and Arow0/Brow0; the distance-twelve leaders had sector-one minima. Verified full ranks gate every emitted rate to [0.275,0.325).")
FILLING = ("Twenty policies cross five matrix-relative-order architectures with four central-C3 occupancy patterns. Each cubic support generates the full SL(2,3) matrix factor through its relative elements. Compile constituent columns only by XOR superposition of verified unit lifts, require connected constituents and exact kernel exclusion through four, and retain rank-stratified pools. Anchored exact decoding through six and eight supplements deeper independent kernel searches. Screen every sector-two singleton and then deepen stabilizer-tested full-code probes equally for all finalists. HX/HZ weights remain 9/12. Combined uncoloured Tanner fingerprints conservatively reject bank equivalents. No old generator or archived filling is used.")
FAMILY_NAME = "r030_SL2F3xC3_12x13_matrix_orders_central_sector2_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 72

def custom_ops(q):
    # SL(2,F3) x C3. Encode (matrix entries a,b,c,d; central t).
    if not valid_q(q):
        raise ValueError("SL(2,F3) x C3 requires order 72")
    mats = [
        (a,b,c,d)
        for a in range(3) for b in range(3)
        for c in range(3) for d in range(3)
        if (a*d-b*c) % 3 == 1
    ]
    assert len(mats) == 24
    els = [m + (t,) for t in range(3) for m in mats]
    positions = {g:i for i,g in enumerate(els)}

    def red(g):
        return tuple(v % 3 for v in g)

    def mul(g, h):
        a,b,c,d,t = red(g)
        e,f,u,v,s = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3,
                (t+s) % 3)

    def inv(g):
        a,b,c,d,t = red(g)
        return (d, (-b) % 3, (-c) % 3, a, (-t) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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


def _dic_packed(rng, G, divisors):
    """Fresh cubic labels with prescribed rotation-pair orders."""
    from math import gcd
    steps = {
        d: [v for v in range(1, 42) if gcd(v, 42) == d]
        for d in (1, 2, 3)
    }
    for restart in range(16):
        used = set()
        row = [None] * len(divisors)
        order = list(range(len(divisors)))
        rng.shuffle(order)
        for j in order:
            for attempt in range(160):
                a, b = rng.randrange(42), rng.randrange(2)
                delta = rng.choice(steps[divisors[j]])
                cell = [(a, b), ((a + delta) % 42, b),
                        (rng.randrange(42), 1 - b)]
                differences = {
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h
                }
                if len(differences) != 6 or differences & used:
                    continue
                if (21, 0) in differences:
                    continue
                row[j] = cell
                used.update(differences)
                break
            if row[j] is None:
                break
        if all(cell is not None for cell in row):
            return [row]
    return None


def _dic_basis(rows):
    basis = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = word
                break
            word ^= previous
    return basis


def _dic_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        previous = basis.get(pivot)
        if previous is None:
            return True
        word ^= previous
    return False


def _dic_probe(cols, rng, trials, best=None, stabilizers=None):
    """Actual kernel witnesses; full-code witnesses exclude stabilizers."""
    best = len(cols) + 1 if best is None else best
    basis = None if stabilizers is None else _dic_basis(stabilizers)
    generators = [] if stabilizers is None else list(stabilizers)
    order = list(range(len(cols)))
    memory = []

    def consider(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if basis is None or _dic_outside(word, basis):
            best = word.bit_count()

    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                dependencies.append(word)
                consider(word)

        dependencies.sort(key=int.bit_count)
        words = sorted(set(dependencies[:40] + memory),
                       key=int.bit_count)[:48]
        # Kernel additions are permitted even when a direction is a
        # stabilizer. Only a non-stabilizer word changes a logical reading.
        for sweep in range(2):
            changed = False
            for i, word in enumerate(words):
                weight = word.bit_count()
                for j, direction in enumerate(words):
                    if i == j:
                        continue
                    combined = word ^ direction
                    if combined and combined.bit_count() < weight:
                        word, weight = combined, combined.bit_count()
                        changed = True
                words[i] = word
                consider(word)
            if not changed:
                break

        discoveries = set(words)
        for i, word in enumerate(words):
            for direction in words[:i]:
                combined = word ^ direction
                consider(combined)
                if combined and combined.bit_count() <= best + 4:
                    discoveries.add(combined)

        # Use the entire information-set basis as descent directions.
        for seed in sorted(discoveries, key=int.bit_count)[:8]:
            word = seed
            for sweep in range(2):
                old_weight = word.bit_count()
                for direction in dependencies:
                    combined = word ^ direction
                    if combined and combined.bit_count() < word.bit_count():
                        word = combined
                if word.bit_count() == old_weight:
                    break
            consider(word)
            discoveries.add(word)

            if generators:
                for sweep in range(2):
                    old_weight = word.bit_count()
                    for direction in generators:
                        combined = word ^ direction
                        if combined and combined.bit_count() < word.bit_count():
                            word = combined
                    if word.bit_count() == old_weight:
                        break
                consider(word)
                discoveries.add(word)

        memory = sorted(discoveries, key=int.bit_count)[:32]
        if best <= 4:
            break
    return best


def _dic_decoder(cols, height):
    """Exact bounded syndrome decoding for cubic constituent columns.

    Branch on an unsatisfied check, then finish with the complete
    singleton/pair table. The caller has excluded kernels through four,
    so every table syndrome has a unique support of size at most two.
    No search-budget cutoff is used unless the caller requests one hit.
    """
    bits = [1 << i for i in range(len(cols))]
    incidence = [[] for _ in range(height)]
    tails = {0: 0}
    for i, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 3
        assert syndrome not in tails
        tails[syndrome] = bits[i]
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length() - 1].append(i)
            value ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            assert value not in tails
            tails[value] = bits[i] | bits[j]

    def decode(syndrome, limit, forbidden=0, cap=0):
        found = set()

        def visit(value, remaining, word):
            if cap and len(found) >= cap:
                return
            if not value:
                found.add(word)
                return
            if value.bit_count() > 3 * remaining:
                return
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not (tail & (word | forbidden))):
                    found.add(word | tail)
                return

            check = (value & -value).bit_length() - 1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & value).bit_count())
            for j in choices:
                if bits[j] & (word | forbidden):
                    continue
                visit(value ^ cols[j], remaining - 1, word | bits[j])
                if cap and len(found) >= cap:
                    return

        visit(syndrome, limit, 0)
        return sorted(found, key=lambda word: (word.bit_count(), word))

    return decode


def _dic_component_pool(rng, G, side, divisors, quota):
    """Independent component optimization, with a larger B search."""
    stub_a = [[[G.e], [G.e]]]
    stub_b = [[[G.e], [G.e], [G.e]]]
    pool = []
    keep = 8 if side == "A" else 16

    for attempt in range(quota):
        cells = _dic_packed(rng, G, divisors)
        if cells is None:
            continue
        A, B = (cells, stub_b) if side == "A" else (stub_a, cells)
        hx, hz, _ = lifted_matrices(G, A, B)
        ar, br, ac, bc = _component_data(hx, hz, G.q)
        rows, cols = (ar, ac) if side == "A" else (br, bc)
        if _component_cycles(rows) or _short_floor(cols) < 5:
            continue
        reading = _kernel_reading(cols, rng, 4 if side == "A" else 3)
        item = {
            "cells": cells, "cols": cols, "d": reading,
            "rank": gf2_rank(rows), "attempt": attempt,
        }
        pool.append(item)
        pool.sort(key=lambda x: (x["d"], -x["rank"],
                                 -x["attempt"]), reverse=True)
        del pool[keep:]

    for item in pool:
        cols = item["cols"]
        decode = _dic_decoder(cols, G.q)
        item["decode"] = decode
        item["clean8"] = True

        # A constituent regular action is transitive inside each block.
        # Thus block-coordinate-zero anchors suffice for constituent
        # tests. This symmetry is NOT assumed for the full quantum code.
        anchors = range(0, len(cols), G.q)
        for bound in (6, 8):
            if any(decode(cols[a], bound - 1, 1 << a, cap=1)
                   for a in anchors):
                item["d"] = min(item["d"], bound)
                item["clean8"] = False
                break
        item["d"] = _dic_probe(
            cols, rng, 40 if side == "A" else 64, item["d"])

    pool.sort(key=lambda x: (
        x["d"], int(x["clean8"]), -x["rank"]), reverse=True)
    return pool[:3 if side == "A" else 4]


def _dic_singleton_readings(item, aa, bb, q):
    """Screen every sector-2 singleton, without a full-code anchor claim.

    X: decode its two q-bit targets in B, through constituent weight five.
    Z: decode its three q-bit targets in A, through constituent weight three.
    Every reported improvement is verified as a non-stabilizer kernel word.
    """
    from itertools import product
    n = item["n"]
    mask = (1 << q) - 1
    n1 = 6 * q
    xc, zc = item["xc"], item["zc"]
    xs, zs = item["xs"], item["zs"]
    dx = dz = n + 1

    def verify(word, cols):
        syndrome = 0
        value = word
        while value:
            bit = value & -value
            syndrome ^= cols[bit.bit_length() - 1]
            value ^= bit
        assert syndrome == 0

    for t in range(q):
        sector_bit = 1 << (n1 + t)
        syndrome = xc[n1 + t]
        choices = [
            bb["decode"]((syndrome >> (j * q)) & mask, 5)
            for j in range(2)
        ]
        if all(choices):
            for v0, v1 in product(*choices):
                word = sector_bit | v0 | (v1 << (3 * q))
                if word.bit_count() < dx and _dic_outside(word, xs):
                    verify(word, xc)
                    dx = word.bit_count()

        syndrome = zc[n1 + t]
        choices = [
            aa["decode"]((syndrome >> (beta * q)) & mask, 3)
            for beta in range(3)
        ]
        if all(choices):
            for compact_words in product(*choices):
                word = sector_bit
                for beta, compact in enumerate(compact_words):
                    word |= (compact & mask) << (beta * q)
                    word |= ((compact >> q) & mask) << ((3 + beta) * q)
                if word.bit_count() < dz and _dic_outside(word, zs):
                    verify(word, zc)
                    dz = word.bit_count()
    return dx, dz


def _dic_check_signature(rows):
    """Permutation-invariant check-intersection fingerprint.

    Equality is conservatively rejected; it is not asserted to prove
    isomorphism. Unequal signatures exclude a check/qubit relabeling.
    """
    size = len(rows)
    adjacency = [0] * size
    overlap = [[0] * size for _ in range(size)]
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = value
            if value:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    signatures = []
    for i, row in enumerate(rows):
        histogram = {}
        for j in range(size):
            if i == j:
                continue
            key = (overlap[i][j],
                   (adjacency[i] & adjacency[j]).bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        signatures.append(
            (row.bit_count(), tuple(sorted(histogram.items()))))
    return tuple(sorted(signatures))


def _dic_fingerprint(item):
    return (
        item["n"], item["k"],
        tuple(sorted((_dic_check_signature(item["hx"]),
                      _dic_check_signature(item["hz"])))),
    )


def _dic_merit(item):
    # Constituent witnesses guide selection but are not automatically
    # claimed to be nontrivial quantum logical representatives.
    d = min(item["da"], item["db"], item["dx"], item["dz"])
    return (item["k"] * d * d / item["n"], d,
            item["da"] + item["db"])


@lru_cache(maxsize=1)
def _dic_design_bank(G):
    """Deterministic, independently optimized structural cards.

    Twenty primary cards use two A architectures and all ten B
    architectures. If sixty slots are requested, the next batches
    use the other four A architectures, rather than orbit copies.
    """
    from itertools import combinations_with_replacement
    a_cards = ((1, 2), (3, 3), (1, 1),
               (2, 3), (2, 2), (1, 3))
    b_cards = tuple(combinations_with_replacement((1, 2, 3), 3))
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    bank, fingerprints = [], set()

    for idx in range(N_SLOTS):
        # A changed stream accompanies the changed construction; no
        # archived labeling is used as an initial code or a template.
        rng = random.Random((BASE_SEED + 137) * 10007 + idx)
        card, batch = idx % 20, idx // 20
        ad = a_cards[(2 * batch + card % 2) % len(a_cards)]
        bd = b_cards[card // 2]
        winner = None

        for retry in range(2):
            ap = _dic_component_pool(
                rng, G, "A", ad, 64 + 24 * retry)
            bp = _dic_component_pool(
                rng, G, "B", bd, 192 + 64 * retry)
            candidates = []
            for aa in ap:
                for bb in bp:
                    A, B = aa["cells"], bb["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (275 * n <= 1000 * k < 325 * n):
                        continue
                    d = min(aa["d"], bb["d"])
                    candidates.append((
                        (k * d * d / n, d), aa, bb, hx, hz, n, k))
            candidates.sort(key=lambda c: c[0], reverse=True)

            finalists = []
            for _, aa, bb, hx, hz, n, k in candidates:
                item = {
                    "A": aa["cells"], "B": bb["cells"],
                    "hx": hx, "hz": hz, "n": n, "k": k,
                    "da": aa["d"], "db": bb["d"],
                }
                fingerprint = _dic_fingerprint(item)
                if fingerprint in fingerprints:
                    continue
                item["fingerprint"] = fingerprint
                item["xc"] = _columns(hz, n)
                item["zc"] = _columns(hx, n)
                item["xs"] = _dic_basis(hx)
                item["zs"] = _dic_basis(hz)
                item["dx"], item["dz"] = _dic_singleton_readings(
                    item, aa, bb, G.q)
                item["dx"] = _dic_probe(
                    item["xc"], rng, 12, item["dx"], hx)
                item["dz"] = _dic_probe(
                    item["zc"], rng, 12, item["dz"], hz)
                finalists.append(item)
                if len(finalists) >= 4:
                    break

            # Give every finalist the same additional full-code budget;
            # retain all earlier lighter witnesses.
            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"], rng, 20, item["dx"], item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"], rng, 20, item["dz"], item["hz"])
                if winner is None or _dic_merit(item) > _dic_merit(winner):
                    winner = item
            if winner is not None and _dic_merit(winner)[1] >= 12:
                break

        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"] + winner["hz"]) <= W_MAX
        assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
        assert all(len(cell) == 3 and len(set(cell)) == 3
                   for M in (winner["A"], winner["B"])
                   for row in M for cell in row)
        fingerprints.add(winner["fingerprint"])
        bank.append((
            tuple(tuple(tuple(cell) for cell in row) for row in winner["A"]),
            tuple(tuple(tuple(cell) for cell in row) for row in winner["B"]),
            winner["k"], _dic_merit(winner),
        ))
    return tuple(bank)


# Active matrix-group construction. Historical Dic21 packers and its
# design bank above are inactive; only their group-independent bit-row
# decoding, basis and witness-search helpers are reused.


def _sl_connected(cols):
    """Connectivity of a constituent using verified column incidences."""
    reached = cols[0]
    occupied = 0
    while True:
        previous = occupied
        for j, syndrome in enumerate(cols):
            if syndrome & reached:
                occupied |= 1 << j
                reached |= syndrome
        if occupied == previous:
            return occupied.bit_count() == len(cols)


def _sl_fingerprint(hx, hz, n):
    """Invariant of the combined UNCOLOURED Tanner graph.

    Unequal signatures exclude graph isomorphism. Equal signatures are
    discarded conservatively, not asserted to prove isomorphism.
    Neither k, X/Z names, nor protograph coordinates enter the invariant.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    degrees = Counter(c.bit_count() for c in _columns(rows, n))
    return (n, tuple(sorted(degrees.items())),
            _dic_check_signature(rows))


@lru_cache(maxsize=1)
def _sl_design_bank(G):
    """Pure deterministic bank of independently optimized order policies."""
    q = G.q
    assert q == 72
    assert A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    mask = (1 << q) - 1
    za, zb = [[[], []]], [[[], [], []]]
    assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
    assert _no_dead_lines([[3, 3]], [[3, 3, 3]])

    # All subsequent constituent compilation is linear superposition
    # of EXACT scaffold-generated one-term regular actions.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                hx, hz, _ = lifted_matrices(G, [[[g], []]], zb)
                rows = [r & mask for r in hx[:q]]
            else:
                hx, hz, _ = lifted_matrices(G, za, [[[g], [], []]])
                rows = [r & mask for r in hz[:q]]
            blocks.append(tuple(_columns(rows, q)))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        width = len(grid[0]) * q
        cols = [0] * width
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for t, syndrome in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+t] ^= syndrome
        rows = [0] * q
        for j, original in enumerate(cols):
            syndrome = original
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length()-1] |= 1 << j
                syndrome ^= bit
        return rows, cols

    # Matrix operations constrain labels, never supply a competing lift.
    matrix_elements = [g for g in G.elements if g[4] == 0]
    matrix_orders = {}
    for g in matrix_elements:
        value = G.e
        for order in range(1, 25):
            value = G.mul(value, g)
            if value == G.e:
                matrix_orders[g] = order
                break
        else:
            raise ValueError("matrix element order failed")

    palettes = {
        order: tuple(g for g in matrix_elements
                     if matrix_orders[g] == order)
        for order in (3, 4, 6)
    }
    recipes = ((4, 3), (4, 6), (3, 3), (3, 6), (6, 6))
    pair_cache = {}

    def generates_matrix(g, h):
        key = (g, h)
        hit = pair_cache.get(key)
        if hit is not None:
            return hit
        reached = {G.e}
        queue = [G.e]
        for value in queue:
            for generator in (g, h):
                product = G.mul(value, generator)
                if product not in reached:
                    reached.add(product)
                    queue.append(product)
        answer = len(reached) == 24
        pair_cache[key] = answer
        return answer

    def draw_grid(local, side, architecture, mode, lane):
        width = 2 if side == "A" else 3
        row = []
        for j in range(width):
            recipe = recipes[
                (architecture + 2*j + int(side == "B") + lane) % 5]
            if mode == 0:
                central = (0, 1, 2)
            elif mode == 1:
                central = ((0, 1, 2) if side == "A" else (0, 0, 1))
            elif mode == 2:
                central = ((0, 0, 1) if side == "A" else (0, 1, 2))
            else:
                central = ((0, 1, 1) if (j + int(side == "B")) % 2
                           else (0, 0, 1))
            # Reverse only the central voltage of this cell, not an
            # automorphism or permutation applied to an emitted code.
            sign = 1 if (j + lane) % 2 == 0 else 2
            for attempt in range(192):
                g = local.choice(palettes[recipe[0]])
                h = local.choice(palettes[recipe[1]])
                if not generates_matrix(g, h):
                    continue
                cell = [
                    G.e,
                    g[:4] + ((sign*central[1]) % 3,),
                    h[:4] + ((sign*central[2]) % 3,),
                ]
                if len(set(cell)) != 3:
                    continue
                differences = {
                    G.mul(x, G.inv(y))
                    for x in cell for y in cell if x != y
                }
                if len(differences) != 6:
                    continue
                if any(G.mul(d, d) == G.e for d in differences):
                    continue
                row.append(cell)
                break
            else:
                raise ValueError("matrix-generating trinomial exhausted")
        return [row]

    def component_key(item):
        # Every reading below comes from an actual constituent witness.
        return (min(item["d"], 16), int(item.get("clean8", False)),
                -item["cycles"])

    def joint_key(item):
        # Classical witnesses guide selection; they are not automatically
        # asserted to survive as nontrivial quantum logicals.
        d = min(item["da"], item["db"], item["dx"], item["dz"], 16)
        return (item["k"]*d*d/item["n"], d,
                item["da"] + item["db"], -item["cycles"])

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        lane = slot // 20
        selected = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED*100003 + 72030091
                + 104729*slot + 7919*batch)

            def component_pool(side):
                strata, seen = {}, set()
                quota = (96 if side == "A" else 160) + 48*batch

                def admit(grid):
                    labels = tuple(sorted(
                        tuple(sorted(cell)) for cell in grid[0]))
                    if labels in seen:
                        return
                    seen.add(labels)
                    rows, cols = compile_component(side, grid)
                    assert all(c.bit_count() == 3 for c in cols)
                    if _short_floor(cols) < 5 or not _sl_connected(cols):
                        return
                    reading = _kernel_reading(cols, local, 3)
                    if reading > len(cols):
                        return
                    rank = gf2_rank(rows)
                    item = dict(
                        cells=grid, cols=cols, rank=rank, d=reading,
                        cycles=_component_cycles(rows))
                    bucket = strata.setdefault(rank, [])
                    bucket.append(item)
                    bucket.sort(key=component_key, reverse=True)
                    del bucket[3:]

                for attempt in range(quota):
                    admit(draw_grid(
                        local, side, architecture, mode, lane))

                # Repairs replace actual matrix/central labels in the
                # limiting constituent rather than transporting a winner.
                for step in range(24):
                    available = sorted(
                        (x for bucket in strata.values() for x in bucket),
                        key=component_key, reverse=True)
                    if not available:
                        admit(draw_grid(
                            local, side, architecture, mode, lane))
                        continue
                    parent = local.choice(available[:4])
                    grid = [[list(c) for c in parent["cells"][0]]]
                    fresh = draw_grid(
                        local, side, architecture, mode, lane)
                    j = local.randrange(len(grid[0]))
                    grid[0][j] = fresh[0][j]
                    if step % 7 == 6:
                        other = (j+1) % len(grid[0])
                        grid[0][other] = fresh[0][other]
                    admit(grid)

                # Preserve measured rank strata before independent deepening.
                pool = sorted(
                    (bucket[0] for bucket in strata.values()),
                    key=component_key, reverse=True)[:8]
                extras = sorted(
                    (x for bucket in strata.values() for x in bucket[1:]),
                    key=component_key, reverse=True)
                pool += extras[:max(0, 8-len(pool))]

                for item in pool:
                    cols = item["cols"]
                    decode = _dic_decoder(cols, q)
                    item["decode"] = decode
                    anchors = range(0, len(cols), q)
                    for anchor in anchors:
                        words = decode(
                            cols[anchor], 5, 1 << anchor, cap=1)
                        if words:
                            item["d"] = min(
                                item["d"], 1 + words[0].bit_count())
                    item["d"] = _dic_probe(
                        cols, local, 16 if side == "A" else 28,
                        item["d"])

                pool.sort(key=component_key, reverse=True)
                pool = pool[:4]
                for item in pool:
                    item["clean8"] = True
                    for anchor in range(0, len(item["cols"]), q):
                        words = item["decode"](
                            item["cols"][anchor], 7,
                            1 << anchor, cap=1)
                        if words:
                            item["clean8"] = False
                            item["d"] = min(
                                item["d"], 1 + words[0].bit_count())
                            break
                    item["d"] = _dic_probe(
                        item["cols"], local,
                        40 if side == "A" else 64, item["d"])
                return sorted(pool, key=component_key, reverse=True)

            ap, bp = component_pool("A"), component_pool("B")
            proposals = []
            for aa in ap:
                for bb in bp:
                    A, B = aa["cells"], bb["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (n == 504 and 275*n <= 1000*k < 325*n):
                        continue
                    ar, br, ac, bc = _component_data(hx, hz, q)
                    assert ac == aa["cols"] and bc == bb["cols"]
                    d = min(aa["d"], bb["d"], 16)
                    proposals.append((
                        (k*d*d/n, d, -aa["cycles"]-bb["cycles"]),
                        aa, bb, hx, hz, n, k))
            proposals.sort(key=lambda x: x[0], reverse=True)

            finalists = []
            for _, aa, bb, hx, hz, n, k in proposals:
                fingerprint = _sl_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                item = dict(
                    A=aa["cells"], B=bb["cells"],
                    hx=hx, hz=hz, n=n, k=k,
                    da=aa["d"], db=bb["d"],
                    cycles=aa["cycles"]+bb["cycles"],
                    fingerprint=fingerprint,
                    xc=_columns(hz, n), zc=_columns(hx, n),
                    xs=_dic_basis(hx), zs=_dic_basis(hz))

                # This helper enumerates ALL q sector-two singletons.
                # Every improvement is syndrome-verified and tested
                # outside the complete matching stabilizer row space.
                item["dx"], item["dz"] = _dic_singleton_readings(
                    item, aa, bb, q)
                item["dx"] = _dic_probe(
                    item["xc"], local, 10, item["dx"], hx)
                item["dz"] = _dic_probe(
                    item["zc"], local, 10, item["dz"], hz)
                if item["dx"] > n or item["dz"] > n:
                    continue
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # No undeepened optimistic finalist wins the final selection.
            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"], local, 24, item["dx"], item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"], local, 24, item["dz"], item["hz"])

            if not finalists:
                continue
            winner = max(finalists, key=joint_key)
            hx, hz, n = winner["hx"], winner["hz"], winner["n"]
            assert css_ok(hx, hz)
            assert n == 504 and winner["k"] >= 144
            assert 275*n <= 1000*winner["k"] < 325*n
            assert max(r.bit_count() for r in hx+hz) == 12
            assert all(len(c) == len(set(c)) == 3
                       for side in ("A", "B")
                       for row in winner[side] for c in row)
            emitted.add(winner["fingerprint"])
            selected = dict(
                cells={
                    side: tuple(tuple(tuple(c) for c in row)
                                for row in winner[side])
                    for side in ("A", "B")},
                k=winner["k"], card=(architecture, mode, lane),
                screen=joint_key(winner))
            break

        # Unsuccessful policies never substitute a parental or bank copy.
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the new matrix group's independently optimized complete grid."""
    selected = _sl_design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh SL2F3xC3 cubic design in target rate cell")
    return {"idx": idx, "cells": selected["cells"],
            "k": selected["k"], "card": selected["card"],
            "screen": selected["screen"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the full jointly selected cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order binomial support profiles preserve the cubic Dic21 distance advantage; every distinct nonuniform-count parent slot read d2s=5, while all five measured parent codes used five trinomials.
# REFUTED: Component-kernel selection alone removes mixed-sector failures on Dic21; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Cubic constituent short-eight and singleton-sector-two selection supplies five Dic21 readings >=12; the measured parent leaders instead read 9,11,12,12,12, with the lighter X witnesses touching both sectors.
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
