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
CONCEPT = ("(C7 x C2^2) semidirect C3, order 84, A=(1,2), B=(1,3): removing central norm mechanisms and selecting cubic constituents against normal-subgroup and dressed logical witnesses should raise five-code fitness above 37.9538 in cell 0.300. This is a falsifiable measured-search hypothesis.")
GROUP = ("L3 jump from Dic21 to the fixed-point-free diagonal C3 action on C7 x C2^2. Multiplication by 2 on C7 and cycling the three nonzero V4 elements both have order three and no nonzero fixed point. Thus the new group has trivial centre, unlike Dic21's centre C2. No centre-trivial order-84 bed is listed; AGL(1,8)'s measured 34.2 motivates testing another fixed-point-free construction without claiming transferred distance.")
SHAPES = ("Keep (1,2)x(1,3) for cell 0.300: n=588, kb=168, guaranteed rate 2/7, unequal constituent lengths 168/252 with 84 checks each, and q<=85 allowed by size. Parent weight-12 minima occupied sector-one Acol/Bcol blocks; measured slots 20 and 22 also retained weight-nine/eleven X witnesses touching Arow0/Brow0. Verified complete ranks enforce 0.275<=k/n<0.325.")
FILLING = ("Twenty policies cross five quotient-C3 occupancy architectures with four normal-coordinate models. Five cubic cells give HX=6+3=9 and HZ=9+3=12. Compile both actions from verified unit lifts, pack directed pair ratios, screen actual C2^2/C7-invariant supports, exactly exclude constituent dependencies through six when possible, and repair independently optimized constituents. Singleton-sector-two decoding, verified embeddings, stabilizer dressings and full-code probes address both parental bottlenecks. Unequal uncoloured incidence fingerprints certify emitted diversity; high sampled readings never certify distance.")
FAMILY_NAME = "r030_C7xV4_semiC3_12x13_normalnorm_cubic_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # (a,u,b) in (C7 x V4):C3.
    # V4 addition is XOR; T cycles 1 -> 2 -> 3 -> 1.
    if not valid_q(q):
        raise ValueError("(C7 x V4):C3 requires order 84")
    els = [(a,u,b) for b in range(3) for u in range(4)
           for a in range(7)]
    action7 = (1,2,4)
    action4 = ((0,1,2,3), (0,2,3,1), (0,3,1,2))

    def red(g):
        return (g[0] % 7, g[1] & 3, g[2] % 3)

    def mul(g, h):
        a,u,b = red(g)
        c,v,d = red(h)
        return ((a + action7[b]*c) % 7,
                u ^ action4[b][v], (b+d) % 3)

    def inv(g):
        a,u,b = red(g)
        bi = (-b) % 3
        return ((-action7[bi]*a) % 7, action4[bi][u], bi)

    def idx(g):
        a,u,b = red(g)
        return a + 7*u + 28*b

    return dict(elements=els, ident=(0,0,0), mul=mul, inv=inv,
                idx=idx, red=red)
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


# The historical Dic21 routines above are inactive. Generic syndrome,
# decoder and stabilizer helpers remain useful with scaffold-derived rows.
_FRO_PATTERNS = ((0,0,1), (0,1,2))
_FRO_CARDS = (
    (0,0,0,0,0),
    (1,1,1,1,1),
    (0,0,0,0,1),
    (0,1,0,1,1),
    (1,1,0,0,0),
)


@lru_cache(maxsize=1)
def _fro_atlas(G):
    """Only verified unit lifts define either component action."""
    q = G.q
    mask = (1 << q)-1
    za, zb = [[[],[]]], [[[],[],[]]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[g],[]]] if side == "A" else za
            B = [[[g],[],[]]] if side == "B" else zb
            hx,hz,_ = lifted_matrices(G,A,B)
            source = hx if side == "A" else hz
            cols = tuple(_columns([r & mask for r in source[:q]],q))
            assert all(c.bit_count() == 1 for c in cols)
            blocks.append(cols)
        atlas[side] = tuple(blocks)
    return atlas


def _fro_columns(G, atlas, side, grid):
    q = G.q
    cols = [0]*(len(grid[0])*q)
    for j,cell in enumerate(grid[0]):
        for g in cell:
            for h,s in enumerate(atlas[side][G.idx(g)]):
                cols[j*q+h] ^= s
    return cols


def _fro_cycles(cols):
    """Actual component Tanner four-cycle count."""
    counts, total = {}, 0
    for s in cols:
        neighbors = []
        while s:
            bit = s & -s
            neighbors.append(bit.bit_length()-1)
            s ^= bit
        for i,a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b,a)
                old = counts.get(key,0)
                total += old
                counts[key] = old+1
    return total


def _fro_syndrome(word, cols):
    s = 0
    while word:
        bit = word & -word
        s ^= cols[bit.bit_length()-1]
        word ^= bit
    return s


def _fro_words(cols, rng, trials, seeds=()):
    """Retain actual nonzero kernel supports, not exclusion sentinels."""
    memory = sorted(
        {w for w in seeds if w and _fro_syndrome(w,cols) == 0},
        key=lambda w:(w.bit_count(),w))[:32]
    order = list(range(len(cols)))
    for trial in range(trials):
        rng.shuffle(order)
        pivots, deps = {}, []
        for c in order:
            s,w = cols[c],1 << c
            while s:
                p = s.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s,w)
                    break
                s ^= old[0]
                w ^= old[1]
            if not s and w:
                deps.append(w)
        words = sorted(set(deps+memory),key=lambda w:(w.bit_count(),w))
        for sweep in range(2):
            donors = words[:28]
            reduced = []
            for w in words:
                weight = w.bit_count()
                for donor in donors:
                    v = w ^ donor
                    vw = v.bit_count()
                    if 0 < vw < weight:
                        w,weight = v,vw
                reduced.append(w)
            words = sorted(set(reduced),key=lambda w:(w.bit_count(),w))
        discoveries = set(words+memory)
        short = sorted(discoveries,key=int.bit_count)[:24]
        for i,w in enumerate(short):
            for v in short[:i]:
                if w ^ v:
                    discoveries.add(w ^ v)
        memory = sorted(discoveries,
                        key=lambda w:(w.bit_count(),w))[:32]
    assert all(w and _fro_syndrome(w,cols) == 0 for w in memory)
    return memory


def _fro_norm_words(cols, fibers, q):
    """True dependencies within normal-subgroup indicator subspaces.

    Gaussian elimination and singleton/pair collisions carry physical
    supports. Absence here excludes only these restricted subspaces.
    """
    found = set()
    width = len(cols)//q
    for partition in fibers.values():
        supports = [f << (j*q) for j in range(width) for f in partition]
        projected = [_fro_syndrome(w,cols) for w in supports]
        seen = {0:0}
        for i,s in enumerate(projected):
            w = supports[i]
            old = seen.get(s)
            if old is None:
                seen[s] = w
            elif w ^ old:
                found.add(w ^ old)
        for i,s in enumerate(projected):
            for j in range(i):
                t,w = s ^ projected[j],supports[i] ^ supports[j]
                old = seen.get(t)
                if old is None:
                    seen[t] = w
                elif w ^ old:
                    found.add(w ^ old)
        pivots,deps = {},[]
        for s,w in zip(projected,supports):
            while s:
                p = s.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s,w)
                    break
                s ^= old[0]
                w ^= old[1]
            if not s and w:
                deps.append(w)
        found.update(deps)
        for i,w in enumerate(deps):
            found.update(w ^ v for v in deps[:i] if w ^ v)
    return sorted(found,key=lambda w:(w.bit_count(),w))[:32]


def _fro_draw(rng, G, side, kinds, mode):
    """Different quotient occupancies and normal-coordinate recipes."""
    for restart in range(24):
        used,row = set(),[]
        for kind in kinds:
            chosen = None
            bs = _FRO_PATTERNS[kind]
            for attempt in range(128):
                if mode == 0:
                    aa = rng.sample(range(7),3)
                elif mode == 1:
                    base,step = rng.randrange(7),rng.randrange(1,7)
                    aa = [(base+t*step) % 7 for t in range(3)]
                    rng.shuffle(aa)
                elif mode == 2:
                    base = rng.randrange(7)
                    other = rng.choice([a for a in range(7) if a != base])
                    aa = [base,base,other]
                    rng.shuffle(aa)
                else:
                    aa = [rng.randrange(7) for _ in range(3)]
                uu = ([0]+rng.sample((1,2,3),2) if mode == 1
                      else [rng.randrange(4) for _ in range(3)])
                cell = list(zip(aa,uu,bs))
                if len(set(cell)) != 3:
                    continue
                anchor = G.inv(cell[0])
                cell = [G.mul(g,anchor) for g in cell]
                g,h = cell[1:]
                if G.mul(g,h) == G.mul(h,g):
                    continue
                differences = {
                    (G.mul(x,G.inv(y)) if side == "A"
                     else G.mul(G.inv(x),y))
                    for x in cell for y in cell if x != y}
                if len(differences) != 6 or differences & used:
                    continue
                if any(G.mul(x,x) == G.e for x in differences):
                    continue
                chosen = cell
                used.update(differences)
                break
            if chosen is None:
                break
            row.append(chosen)
        if len(row) == len(kinds):
            return [row]
    return None


def _fro_pool(rng, G, atlas, fibers, side, kinds, mode, batch):
    q = G.q
    raw, seen = [],set()
    quota = (72 if side == "A" else 112)+48*batch

    def labels(grid):
        return tuple(tuple(sorted(c)) for c in grid[0])

    def admit(grid):
        if grid is None:
            return
        sig = labels(grid)
        if sig in seen:
            return
        seen.add(sig)
        cols = _fro_columns(G,atlas,side,grid)
        assert all(c.bit_count() == 3 for c in cols)
        cycles = _fro_cycles(cols)
        if cycles or _short_floor(cols) < 5:
            return
        seeds = _fro_norm_words(cols,fibers,q)
        norm = min((w.bit_count() for w in seeds),default=len(cols)+1)
        if norm <= 6:
            return
        item = dict(cells=grid,cols=cols,seeds=seeds,
                    rank=gf2_rank(cols),cycles=cycles,
                    norm=norm,label=sig)
        raw.append(item)
        # This cap is a structural selection target, not a found distance.
        raw.sort(key=lambda x:(-min(x["norm"],16),
                               -x["rank"],x["label"]))
        del raw[14:]

    for attempt in range(quota):
        admit(_fro_draw(rng,G,side,kinds,mode))

    def merit(item):
        return (min(item["d"],16),
                -sum(w.bit_count() == item["d"] for w in item["words"]),
                item["rank"])

    pool = []
    for item in raw:
        item["words"] = _fro_words(item["cols"],rng,3,item["seeds"])
        if not item["words"]:
            continue
        item["d"] = item["words"][0].bit_count()
        pool.append(item)
    pool.sort(key=merit,reverse=True)
    pool = pool[:6]

    for step in range(16):
        if not pool:
            break
        parent = rng.choice(pool[:3])
        grid = [[list(c) for c in parent["cells"][0]]]
        replacement = _fro_draw(rng,G,side,kinds,mode)
        if replacement is None:
            continue
        pressure = [
            1+sum(((w >> (j*q)) & ((1 << q)-1)).bit_count()
                  for w in parent["words"][:8])
            for j in range(len(kinds))]
        j = rng.choices(range(len(kinds)),weights=pressure,k=1)[0]
        grid[0][j] = replacement[0][j]
        sig = labels(grid)
        if sig in seen:
            continue
        seen.add(sig)
        cols = _fro_columns(G,atlas,side,grid)
        if _fro_cycles(cols) or _short_floor(cols) < 5:
            continue
        seeds = _fro_norm_words(cols,fibers,q)
        # Old witnesses survive only if verified on the changed matrix.
        seeds.extend(w for w in parent["words"]
                     if _fro_syndrome(w,cols) == 0)
        words = _fro_words(cols,rng,3,seeds)
        if not words:
            continue
        pool.append(dict(cells=grid,cols=cols,words=words,
                         d=words[0].bit_count(),rank=gf2_rank(cols)))
        pool.sort(key=merit,reverse=True)
        del pool[6:]

    for item in pool:
        item["words"] = _fro_words(item["cols"],rng,12,item["words"])
        item["d"] = min(item["d"],item["words"][0].bit_count())
    pool.sort(key=merit,reverse=True)
    pool = pool[:3 if side == "A" else 4]

    for item in pool:
        cols = item["cols"]
        decode = _dic_decoder(cols,q)
        item["decode"] = decode
        # Complete constituent test through six: regular-action
        # transitivity permits one anchor per protograph column block.
        for a in range(0,len(cols),q):
            hits = decode(cols[a],5,1 << a,cap=1)
            if hits:
                witness = hits[0] | (1 << a)
                assert _fro_syndrome(witness,cols) == 0
                item["words"].append(witness)
                item["d"] = min(item["d"],witness.bit_count())
        item["words"] = _fro_words(cols,rng,24,item["words"])
        item["d"] = min(item["d"],item["words"][0].bit_count())
    pool.sort(key=merit,reverse=True)
    return pool


def _fro_dressed(item, aa, bb, q):
    """Verify constituent embeddings and their nontrivial logical classes."""
    mask = (1 << q)-1
    best = item["n"]+1
    for side,seeds,checks,stabilizers in (
            ("A",aa["words"],item["hx"],item["hz"]),
            ("B",bb["words"],item["hz"],item["hx"])):
        span = _dic_basis(stabilizers)
        for seed in sorted(set(seeds),key=int.bit_count)[:8]:
            for fixed in range(3 if side == "A" else 2):
                word = (sum(((seed >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(2))
                        if side == "A" else seed << (fixed*3*q))
                assert not any((word & row).bit_count() & 1
                               for row in checks)
                if not _dic_outside(word,span):
                    continue
                best = min(best,word.bit_count())
                for sweep in range(3):
                    chosen,weight = word,word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(),row)
                         for row in stabilizers),
                        key=lambda x:x[0])[:12]
                    for _,row in nearby:
                        trial = word ^ row
                        tw = trial.bit_count()
                        if tw < weight:
                            chosen,weight = trial,tw
                    for i,(_,row) in enumerate(nearby):
                        for _,other in nearby[:i]:
                            trial = word ^ row ^ other
                            tw = trial.bit_count()
                            if tw < weight:
                                chosen,weight = trial,tw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best,weight)
    return best


def _fro_signature(hx, hz, n):
    """Uncoloured incidence invariant; inequality proves nonisomorphism."""
    from collections import Counter
    rows = list(hx)+list(hz)
    incidents = [[] for _ in range(n)]
    for r,source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length()-1].append(r)
            source ^= bit
    adjacency = [Counter() for _ in rows]
    for neighbors in incidents:
        for i,a in enumerate(neighbors):
            for b in neighbors[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1
    profiles = []
    for r,neighbors in enumerate(adjacency):
        square = {}
        for s,m in neighbors.items():
            for t,v in adjacency[s].items():
                square[t] = square.get(t,0)+m*v
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r,0),
            sum(neighbors.get(t,0)*v for t,v in square.items()),
            sum(v*v for v in square.values()),
            tuple(sorted(Counter(square.values()).items()))))
    return (n,tuple(sorted(Counter(map(len,incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


def _fro_small_logical(cols, stabilizers):
    """Exact full-code logical existence through four."""
    span = _dic_basis(stabilizers)
    seen = {0:0}
    bits = [1 << i for i in range(len(cols))]
    for i,s in enumerate(cols):
        w = bits[i]
        old = seen.get(s)
        if old is None:
            seen[s] = w
        elif _dic_outside(w ^ old,span):
            return True
    for i,s in enumerate(cols):
        for j in range(i):
            t,w = s ^ cols[j],bits[i] | bits[j]
            old = seen.get(t)
            if old is None:
                seen[t] = w
            elif _dic_outside(w ^ old,span):
                return True
    return False


@lru_cache(maxsize=1)
def _fro_bank(G):
    """Pure deterministic bank of fresh, incidence-distinct design cards."""
    assert G.q == 84 and A_SHAPE == (1,2) and B_SHAPE == (1,3)
    q = G.q
    atlas = _fro_atlas(G)
    fibers = {
        4: tuple(sum(1 << G.idx((a,u,b)) for u in range(4))
                 for b in range(3) for a in range(7)),
        7: tuple(sum(1 << G.idx((a,u,b)) for a in range(7))
                 for b in range(3) for u in range(4)),
    }
    assert _grid_weights([[3,3]],[[3,3,3]]) == 12
    assert _no_dead_lines([[3,3]],[[3,3,3]])
    bank,emitted = [],set()

    def merit(item):
        d = min(item["da"],item["db"],item["dx"],item["dz"],16)
        return (item["k"]*d*d/item["n"],d,
                item["da"]+item["db"])

    for slot in range(N_SLOTS):
        architecture,mode = divmod(slot % 20,4)
        policy = _FRO_CARDS[architecture]
        rng = random.Random(BASE_SEED*100003+8407123+104729*slot)
        winner = None
        for batch in range(3):
            ap = _fro_pool(rng,G,atlas,fibers,"A",
                           policy[:2],mode,batch)
            bp = _fro_pool(rng,G,atlas,fibers,"B",
                           policy[2:],mode,batch)
            pairs = sorted(
                ((min(a["d"],b["d"]),a["d"]+b["d"],i,j,a,b)
                 for i,a in enumerate(ap) for j,b in enumerate(bp)),
                key=lambda x:x[:4],reverse=True)
            finalists = []
            for _,_,_,_,aa,bb in pairs:
                A,B = aa["cells"],bb["cells"]
                hx,hz,n = lifted_matrices(G,A,B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 588 and 275*n <= 1000*k < 325*n):
                    continue
                _,_,ac,bc = _component_data(hx,hz,q)
                assert ac == aa["cols"] and bc == bb["cols"]
                xc,zc = _columns(hz,n),_columns(hx,n)
                if _fro_small_logical(xc,hx):
                    continue
                if _fro_small_logical(zc,hz):
                    continue
                fp = _fro_signature(hx,hz,n)
                if fp in emitted:
                    continue
                item = dict(A=A,B=B,hx=hx,hz=hz,n=n,k=k,
                            da=aa["d"],db=bb["d"],xc=xc,zc=zc,
                            xs=_dic_basis(hx),zs=_dic_basis(hz),
                            fingerprint=fp)
                dx,dz = _dic_singleton_readings(item,aa,bb,q)
                dressed = _fro_dressed(item,aa,bb,q)
                item["dx"] = min(dx,dressed)
                item["dz"] = min(dz,dressed)
                # The shared dressed minimum is conservative for selection;
                # full-type readings below only use verified logical words.
                item["dx"] = _dic_probe(xc,rng,6,item["dx"],hx)
                item["dz"] = _dic_probe(zc,rng,6,item["dz"],hz)
                finalists.append(item)
                if len(finalists) == 3:
                    break
            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"],rng,16,item["dx"],item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"],rng,16,item["dz"],item["hz"])
                if min(item["dx"],item["dz"]) > item["n"]:
                    continue
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 12:
                break

        if winner is None:
            bank.append(None)
            continue
        A,B = winner["A"],winner["B"]
        assert css_ok(winner["hx"],winner["hz"])
        assert max(r.bit_count() for r in
                   winner["hx"]+winner["hz"]) == 12
        assert winner["k"] >= 168
        assert all(len(c) == len(set(c)) == 3
                   for grid in (A,B) for row in grid for c in row)
        emitted.add(winner["fingerprint"])
        bank.append((
            tuple(tuple(tuple(c) for c in row) for row in A),
            tuple(tuple(tuple(c) for c in row) for row in B),
            winner["k"],merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this group's independently optimized complete filling."""
    selected = _fro_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh target-rate Frobenius filling passed")
    A,B,k,score = selected
    return {"idx":idx,"cells":{"A":A,"B":B},"k":k,"screen":score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order binomial support profiles preserve the cubic Dic21 distance advantage; every distinct nonuniform-count parent slot read d2s=5, while all five measured parent codes used five trinomials.
# REFUTED: Component-kernel selection alone removes mixed-sector failures on Dic21; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Exact constituent short-dependency tests and singleton-sector-two screening ensure five Dic21 readings >=12; current measured slots 20 and 22 instead retained mixed-sector X readings nine and eleven, and five-code fitness was 37.9538.
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
