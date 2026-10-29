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
CONCEPT = ("Dic12, order 48, A=(1,2), B=(2,5): controlled-rank quartic A constituents coupled to two-row degree-three/four B constituents should exceed five-code fitness 43.102 in cell 0.300; measured distance searches can refute this.")
GROUP = ("L2 resize within the unchanged dicyclic construction rule, from Dic21 order 84 to Dic12 order 48. The parent measured five k=176, d30s=12 leaders and fitness 43.102. Its order does not fit the new shape; order 48 gives n=576. Transfer of distance-twelve performance is unproved.")
SHAPES = ("New (1,2)x(2,5): n=576, kb=144, guaranteed rate 1/4 and size-law q<=49. A has 96 variables/48 checks and B has 240 variables/96 checks. Target membership requires measured rank surplus, so accept only 0.275<=k/n<0.325. Parent minima occupied sector-one Acol/Bcol blocks, especially Acol1 and Bcol1; the second B row changes that constituent bottleneck rather than transposing an archived code.")
FILLING = ("Twenty primary tasks cross five distinct B count profiles with four normal-C3 quotient-rank preferences. Both A cells are quartic; each B row totals eight and each B column has three or four terms, giving HX/HZ maxima twelve. Enumerate quotient support pairs using scaffold-derived atoms, lift fresh normal coordinates, retain target A rank deficits and full-row-rank B constituents, and rescreen actual norm, information-set and bounded short-dependency witnesses. Exact complete ranks, logical-four exclusion, stabilizer-tested embeddings and equal deeper logical budgets select outputs. Unequal uncoloured Tanner invariants certify bank diversity; sampled readings and restricted exclusions never certify quantum distance.")
FAMILY_NAME = "r030_dic12_12x25_quartic_quotientrank_twoRowB_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Same dicyclic rule: Dic12, x^24=1, y^2=x^12.
    if q != 48:
        raise ValueError("Dic12 has order 48")
    els = [(a, b) for b in range(2) for a in range(24)]

    def red(g):
        a, b = g
        return ((a + 12 * (b // 2)) % 24, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 12*b*d) % 24,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 24, 0) if b == 0 else ((a + 12) % 24, 1)

    def idx(g):
        a, b = red(g)
        return 24*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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
    """Extract arbitrary-shaped constituents from the verified LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = [
        sum(((hx[(i*cb)*q+t] >> (j*cb*q)) & mask) << (j*q)
            for j in range(ca))
        for i in range(ra) for t in range(q)
    ]
    br = [row & ((1 << (cb*q))-1) for row in hz[:rb*q]]
    return ar, br, _columns(ar, ca*q), _columns(br, cb*q)


from functools import lru_cache


def _dic_packed(rng, G, divisors):
    """Fresh cubic labels with prescribed rotation-pair orders."""
    from math import gcd
    steps = {
        d: [v for v in range(1, 42) if gcd(v, 42) == d]
        for d in (1, 2, 3, 6)
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


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure deterministic bank; idx=-1 shares verified compilation."""
    if idx >= 0:
        selected = NEW_CTX(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh norm/multisector-screened cubic card")
        A, B, k, score = selected
        return {
            "idx": idx, "k": k, "screen": score,
            "cells": {
                "A": [[list(c) for c in row] for row in A],
                "B": [[list(c) for c in row] for row in B]}}

    from itertools import combinations, combinations_with_replacement

    assert G.q == 84
    assert A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    q, mask = G.q, (1 << G.q) - 1
    zero_a, zero_b = [[[], []]], [[[], [], []]]
    widths = {"A": 2, "B": 3}

    # All parity-matrix atoms come from the scaffold, not a hand lift.
    atlas = {}
    for side in ("A", "B"):
        atoms = []
        for g in G.elements:
            A, B = zero_a, zero_b
            if side == "A":
                A = [[[g], []]]
            else:
                B = [[[g], [], []]]
            hx, hz, _ = lifted_matrices(G, A, B)
            _, _, ac, bc = _component_data(hx, hz, q)
            atoms.append(tuple((ac if side == "A" else bc)[:q]))
        atlas[side] = tuple(atoms)

    def compile_component(side, cells):
        cols = [0] * (widths[side] * q)
        for j, cell in enumerate(cells[0]):
            for g in cell:
                atom = atlas[side][G.idx(g)]
                for h, syndrome in enumerate(atom):
                    cols[j*q+h] ^= syndrome
        return cols

    def syndrome_of(word, cols):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    # These maps transport SUPPORTS only. Their compatibility with the
    # scaffold-generated atoms is checked explicitly below.
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    permutations, check_maps, word_maps = {}, {}, {}
    for side in ("A", "B"):
        ps = []
        for g in G.elements:
            ps.append(tuple(
                G.idx(G.mul(h, g) if side == "A" else G.mul(g, h))
                for h in G.elements))
        permutations[side] = tuple(ps)
        check_maps[side] = tuple(
            tuple(1 << h for h in p) for p in ps)
        word_maps[side] = tuple(
            tuple(1 << (j*q+p[h])
                  for j in range(widths[side]) for h in range(q))
            for p in ps)
        for generator in ((1, 0), (0, 1), (21, 0)):
            t = G.idx(generator)
            p, moved = ps[t], check_maps[side][t]
            for atom in atlas[side]:
                assert all(moved[s.bit_length()-1] == atom[p[h]]
                           for h, s in enumerate(atom))

    def move(word, maps):
        result = 0
        while word:
            bit = word & -word
            result ^= maps[bit.bit_length()-1]
            word ^= bit
        return result

    canonical_cache = {"A": {}, "B": {}}

    def canonical(side, syndrome):
        cache = canonical_cache[side]
        old = cache.get(syndrome)
        if old is not None:
            return old
        if syndrome == 0:
            return 0, G.idx(G.e)
        remaining, winner = syndrome, None
        while remaining:
            bit = remaining & -remaining
            t = inverse[bit.bit_length()-1]
            candidate = move(syndrome, check_maps[side][t])
            key = (candidate, t)
            if winner is None or key < winner:
                winner = key
            remaining ^= bit
        if len(cache) > 250000:
            cache.clear()
        cache[syndrome] = winner
        return winner

    z = (21, 0)
    covered, orbits = set(), []
    for g in G.elements:
        a = G.idx(g)
        if a in covered:
            continue
        b = G.idx(G.mul(z, g))
        assert a != b
        orbit = tuple(sorted((a, b)))
        covered.update(orbit)
        orbits.append(orbit)
    assert len(orbits) == 42
    output_masks = tuple((1 << a) | (1 << b) for a, b in orbits)

    def norm_completion_reading(cols):
        """Actual restricted witnesses, not full distance certification.

        Folded six-supports are enumerated using regular-action anchors.
        Every orientation and <=2 norm-pair correction is tested for each
        such support. This covers physical completions through ten of
        those projections, not projections of weights eight or ten.
        """
        folded, norms, representatives, norm_syndromes = [], [], [], []
        for block in range(len(cols)//q):
            for a, b in orbits:
                a, b = block*q+a, block*q+b
                s = cols[a] ^ cols[b]
                compact = 0
                for r, output in enumerate(output_masks):
                    part = s & output
                    assert part in (0, output)
                    if part:
                        compact |= 1 << r
                folded.append(compact)
                norms.append((1 << a) | (1 << b))
                representatives.append(1 << a)
                norm_syndromes.append(s)
        floor = _short_floor(folded)
        if floor < 5:
            # Exact existence in the restricted norm subspace.
            return 2*floor, ()
        assert all(s.bit_count() == 3 for s in folded)
        decode = _dic_decoder(folded, len(orbits))
        corrections = {0: 0}
        for i, s in enumerate(norm_syndromes):
            assert s not in corrections
            corrections[s] = norms[i]
        for i, s in enumerate(norm_syndromes):
            for j in range(i):
                key = s ^ norm_syndromes[j]
                assert key not in corrections
                corrections[key] = norms[i] | norms[j]

        best, witnesses, seen = len(cols)+1, set(), set()
        for anchor in range(0, len(folded), len(orbits)):
            for tail in decode(
                    folded[anchor], 5, 1 << anchor):
                projection = tail | (1 << anchor)
                if projection in seen:
                    continue
                seen.add(projection)
                assert projection.bit_count() == 6
                selected, remaining = [], projection
                while remaining:
                    bit = remaining & -remaining
                    selected.append(bit.bit_length()-1)
                    remaining ^= bit
                norm_word = 0
                for c in selected:
                    norm_word ^= norms[c]
                assert syndrome_of(norm_word, cols) == 0
                witnesses.add(norm_word)
                best = min(best, norm_word.bit_count())

                word = 0
                for c in selected:
                    word ^= representatives[c]
                residual = syndrome_of(word, cols)
                for orientation in range(64):
                    if orientation:
                        t = (orientation & -orientation).bit_length()-1
                        c = selected[t]
                        word ^= norms[c]
                        residual ^= norm_syndromes[c]
                    correction = corrections.get(residual)
                    if correction is None:
                        continue
                    physical = word ^ correction
                    assert physical and syndrome_of(physical, cols) == 0
                    weight = physical.bit_count()
                    assert weight <= 10
                    witnesses.add(physical)
                    return min(best, weight), tuple(sorted(
                        witnesses, key=lambda w: (w.bit_count(), w))[:24])
        return best, tuple(sorted(
            witnesses, key=lambda w: (w.bit_count(), w))[:24])

    def component_pool(local, side, divisors, quota):
        """Independent fresh voltages, with restricted physical rescreening."""
        raw, seen = [], set()
        keep = 16 if side == "A" else 28
        for attempt in range(quota):
            cells = _dic_packed(local, G, divisors)
            if cells is None:
                continue
            label = tuple(tuple(sorted(c)) for c in cells[0])
            if label in seen:
                continue
            seen.add(label)
            cols = compile_component(side, cells)
            rows = _columns(cols, q)
            if _component_cycles(rows) or _short_floor(cols) < 5:
                continue
            d = _kernel_reading(cols, local, 2)
            raw.append(dict(
                cells=cells, cols=cols, d=d, rank=gf2_rank(cols),
                attempt=attempt))
            raw.sort(key=lambda a: (
                a["d"], -a["rank"], -a["attempt"]), reverse=True)
            del raw[keep:]

        pool = []
        for item in raw:
            cols = item["cols"]
            decode = _dic_decoder(cols, q)
            short = False
            for bound in (6, 8):
                for anchor in range(0, len(cols), q):
                    tails = decode(cols[anchor], bound-1,
                                   1 << anchor, cap=1)
                    if tails:
                        word = tails[0] | (1 << anchor)
                        assert syndrome_of(word, cols) == 0
                        item["d"] = min(item["d"], word.bit_count())
                        short = True
                        break
                if short:
                    break
            if short:
                continue
            norm_d, seeds = norm_completion_reading(cols)
            item["d"] = min(item["d"], norm_d)
            item["seeds"] = seeds
            item["decode"] = decode
            item["side"] = side
            item["d"] = _dic_probe(
                cols, local, 24 if side == "A" else 40, item["d"])
            pool.append(item)
        pool.sort(key=lambda a: (
            min(a["d"], 12), a["d"], -a["rank"]), reverse=True)
        return pool[:4 if side == "A" else 6]

    def make_query(item):
        """Exact bounded decoding using regular-action triple tails.

        Through-eight exclusion implies uniqueness of supports through
        three for a fixed syndrome. All orbit representatives of singleton,
        pair and triple supports enter the table, not sampled triples.
        """
        if "query" in item:
            return item["query"]
        side, cols = item["side"], item["cols"]
        bits = [1 << i for i in range(len(cols))]
        tails = {0: 0}

        def record(syndrome, word):
            key, t = canonical(side, syndrome)
            normalized = move(word, word_maps[side][t])
            old = tails.get(key)
            if old is None:
                tails[key] = normalized
            else:
                assert old == normalized, "constituent-six screen mismatch"

        for a in range(0, len(cols), q):
            record(cols[a], bits[a])
            for b in range(len(cols)):
                if b != a:
                    record(cols[a] ^ cols[b], bits[a] | bits[b])
            for b in range(len(cols)-1):
                if b == a:
                    continue
                partial = cols[a] ^ cols[b]
                word = bits[a] | bits[b]
                for c in range(b+1, len(cols)):
                    if c != a:
                        record(partial ^ cols[c], word | bits[c])

        incidence = [[] for _ in range(q)]
        for i, s in enumerate(cols):
            while s:
                bit = s & -s
                incidence[bit.bit_length()-1].append(i)
                s ^= bit
        memo = {}

        def query(syndrome, limit):
            key, transport = canonical(side, syndrome)
            cache_key = (key, limit)
            found = memo.get(cache_key)
            if found is None:
                answers = set()

                def visit(value, remaining, word):
                    if not value:
                        answers.add(word)
                        return
                    if value.bit_count() > 3*remaining:
                        return
                    if remaining <= 3:
                        target, t = canonical(side, value)
                        tail = tails.get(target)
                        if tail is None or tail.bit_count() > remaining:
                            return
                        tail = move(tail, word_maps[side][inverse[t]])
                        if not tail & word:
                            answers.add(word | tail)
                        return
                    check = (value & -value).bit_length()-1
                    for c in incidence[check]:
                        if not word & bits[c]:
                            visit(value ^ cols[c], remaining-1,
                                  word | bits[c])

                visit(key, limit, 0)
                found = tuple(sorted(
                    answers, key=lambda w: (w.bit_count(), w)))
                assert all(syndrome_of(w, cols) == key for w in found)
                memo[cache_key] = found
            return tuple(move(w, word_maps[side][inverse[transport]])
                         for w in found)

        item["query"] = query
        return query

    def mixed_x_reading(item, bb):
        """Actual X witnesses from bounded sector-2 supports.

        Pairs use total constituent budget eight. Triples use budget six,
        covering all weight-nine X witnesses with three sector-2 qubits.
        Singleton searches permit asymmetric completions through seven.
        Larger sector-2 supports and triple weight-eleven completions are
        not exhaustively screened here.
        """
        query = make_query(bb)
        xc, span = item["xc"], item["xs"]
        sector = xc[6*q:]
        best = item["dx"]

        def scan(positions, budget, small_cap, other_cap):
            nonlocal best
            s, sector_word = 0, 0
            for t in positions:
                s ^= sector[t]
                sector_word |= 1 << (6*q+t)
            targets = (s & mask, (s >> q) & mask)
            lower = []
            for target in targets:
                bound = (target.bit_count()+2)//3
                if (bound-target.bit_count()) & 1:
                    bound += 1
                lower.append(bound)
            if sum(lower) > budget:
                return
            for first in (0, 1):
                second = 1-first
                limit = min(small_cap, budget-lower[second])
                if limit < 0:
                    continue
                for left in query(targets[first], limit):
                    remaining = min(other_cap, budget-left.bit_count())
                    if remaining < lower[second]:
                        continue
                    for right in query(targets[second], remaining):
                        parts = (left, right) if first == 0 else (right, left)
                        word = sector_word | parts[0] | (parts[1] << (3*q))
                        weight = word.bit_count()
                        if weight >= best or not _dic_outside(word, span):
                            continue
                        assert syndrome_of(word, xc) == 0
                        best = weight

        for t in range(q):
            scan((t,), 10, 5, 7)
            if best <= 9:
                return best
        for positions in combinations(range(q), 2):
            scan(positions, 8, 4, 6)
            if best <= 10:
                return best
        for positions in combinations(range(q), 3):
            scan(positions, 6, 3, 5)
            if best <= 9:
                return best
        return best

    def fingerprint(hx, hz, n):
        # Ignore X/Z colouring and colour-dependent rank information.
        return n, _dic_check_signature(list(hx)+list(hz))

    # Exclusion templates only: never used as proposals or emitted cells.
    parental_measured = (
        (((37,0),(6,1),(17,1)), ((25,0),(5,1),(30,1)),
         ((2,0),(3,1),(26,1)), ((39,0),(3,1),(4,1)),
         ((20,0),(37,0),(15,1))),
        (((24,0),(35,0),(38,1)), ((1,0),(38,0),(0,1)),
         ((20,0),(9,1),(40,1)), ((17,0),(20,1),(37,1)),
         ((13,0),(35,0),(0,1))),
        (((0,0),(31,0),(17,1)), ((14,0),(1,1),(6,1)),
         ((14,0),(8,1),(21,1)), ((0,0),(32,0),(7,1)),
         ((33,0),(35,0),(30,1))),
        (((9,0),(20,0),(35,1)), ((14,0),(27,1),(32,1)),
         ((7,0),(41,0),(17,1)), ((10,0),(37,0),(1,1)),
         ((3,0),(6,0),(7,1))),
        (((14,0),(28,1),(41,1)), ((40,0),(20,1),(21,1)),
         ((7,0),(17,1),(26,1)), ((14,0),(17,1),(32,1)),
         ((9,0),(11,1),(14,1))),
    )
    emitted = set()
    for cells in parental_measured:
        A, B = [[list(c) for c in cells[:2]]], [[list(c) for c in cells[2:]]]
        hx, hz, n = lifted_matrices(G, A, B)
        emitted.add(fingerprint(hx, hz, n))

    b_cards = tuple(combinations_with_replacement((1, 2, 3, 6), 3))
    assert len(b_cards) == 20
    a_cards = ((1, 1), (1, 2), (2, 2))
    bank = []

    def merit(item):
        # Twelve is a conservative optimization cap, not a certificate.
        d = min(item["da"], item["db"], item["dx"], item["dz"], 12)
        return (item["k"]*d*d/item["n"], d,
                item["da"]+item["db"])

    for slot in range(N_SLOTS):
        local = random.Random(BASE_SEED*100003 + 84031091 + 104729*slot)
        ad = a_cards[(slot//20) % len(a_cards)]
        bd = b_cards[slot % 20]
        winner = None
        for batch in range(2):
            ap = component_pool(local, "A", ad, 112+48*batch)
            bp = component_pool(local, "B", bd, 256+96*batch)
            proposals = []
            for aa in ap:
                for bb in bp:
                    hx, hz, n = lifted_matrices(G, aa["cells"], bb["cells"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (275*n <= 1000*k < 325*n):
                        continue
                    _, _, ac, bc = _component_data(hx, hz, q)
                    assert ac == aa["cols"] and bc == bb["cols"]
                    fp = fingerprint(hx, hz, n)
                    if fp in emitted:
                        continue
                    d = min(aa["d"], bb["d"], 12)
                    proposals.append((k*d*d, aa, bb, hx, hz, n, k, fp))
            proposals.sort(key=lambda p: p[0], reverse=True)

            finalists = []
            for _, aa, bb, hx, hz, n, k, fp in proposals:
                item = dict(
                    A=aa["cells"], B=bb["cells"], hx=hx, hz=hz,
                    n=n, k=k, da=aa["d"], db=bb["d"], fp=fp,
                    xc=_columns(hz, n), zc=_columns(hx, n),
                    xs=_dic_basis(hx), zs=_dic_basis(hz))
                item["dx"], item["dz"] = _dic_singleton_readings(
                    item, aa, bb, q)

                # Only syndrome- and stabilizer-tested embeddings lower
                # full-code readings; constituent words are not assumed
                # automatically to represent nontrivial logical classes.
                for side, component in (("A", aa), ("B", bb)):
                    for seed in component["seeds"][:12]:
                        for fixed in range(3 if side == "A" else 2):
                            if side == "A":
                                word = sum(
                                    ((seed >> (j*q)) & mask)
                                    << ((3*j+fixed)*q) for j in range(2))
                                cols, span, field = item["zc"], item["zs"], "dz"
                            else:
                                word = seed << (fixed*3*q)
                                cols, span, field = item["xc"], item["xs"], "dx"
                            if (syndrome_of(word, cols) == 0
                                    and _dic_outside(word, span)):
                                item[field] = min(item[field], word.bit_count())

                if min(item["da"], item["db"], item["dx"], item["dz"]) >= 12:
                    item["dx"] = mixed_x_reading(item, bb)
                item["dx"] = _dic_probe(
                    item["xc"], local, 12, item["dx"], hx)
                item["dz"] = _dic_probe(
                    item["zc"], local, 12, item["dz"], hz)
                finalists.append(item)
                if len(finalists) >= 3:
                    break

            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"], local, 20, item["dx"], item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"], local, 20, item["dz"], item["hz"])
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 12:
                break

        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["n"] == 588 and winner["k"] >= 168
        assert max(r.bit_count()
                   for r in winner["hx"]+winner["hz"]) <= W_MAX
        assert all(len(c) == len(set(c)) == 3
                   for grid in (winner["A"], winner["B"])
                   for row in grid for c in row)
        emitted.add(winner["fp"])
        bank.append((
            tuple(tuple(tuple(c) for c in row) for row in winner["A"]),
            tuple(tuple(tuple(c) for c in row) for row in winner["B"]),
            winner["k"], merit(winner)))
    return tuple(bank)


# The historical order-84 optimizers above are inactive. The following
# bank and NEW_CTX replace them for the new shape and order.
_S48_B_COUNTS = (
    ((2,2,1,1,2), (1,1,2,2,2)),
    ((3,2,1,0,2), (0,1,2,3,2)),
    ((3,3,0,0,2), (0,0,3,3,2)),
    ((3,1,1,0,3), (0,2,2,3,1)),
    ((1,1,1,1,4), (2,2,2,2,0)),
)


def _s48_syndrome(word, cols):
    value = 0
    while word:
        bit = word & -word
        value ^= cols[bit.bit_length()-1]
        word ^= bit
    return value


def _s48_words(cols, local, trials, seeds=()):
    """Retain verified kernel words; sampling misses are not exclusions."""
    memory = sorted(
        {w for w in seeds if w and not _s48_syndrome(w, cols)},
        key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))
    for _ in range(trials):
        local.shuffle(order)
        pivots, deps = {}, []
        for c in order:
            value, word = cols[c], 1 << c
            while value:
                p = value.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value and word:
                deps.append(word)
        words = sorted(set(deps+memory),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        for sweep in range(2):
            donors = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
            discoveries.update(words[:32])
        short = sorted(discoveries, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            discoveries.update(word ^ other for other in short[:i]
                               if word ^ other)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
    assert all(w and not _s48_syndrome(w, cols) for w in memory)
    return tuple(memory)


def _s48_decoder(cols, height):
    """Exact bounded constituent decoding after exclusion through four."""
    bits = [1 << j for j in range(len(cols))]
    incidence = [[] for _ in range(height)]
    tails = {0: 0}
    degree = max(c.bit_count() for c in cols)
    for j, value in enumerate(cols):
        assert value not in tails
        tails[value] = bits[j]
        pending = value
        while pending:
            bit = pending & -pending
            incidence[bit.bit_length()-1].append(j)
            pending ^= bit
    for j, value in enumerate(cols):
        for h in range(j):
            syndrome = value ^ cols[h]
            assert syndrome not in tails
            tails[syndrome] = bits[j] | bits[h]

    def decode(value, limit, forbidden=0, cap=1):
        found = set()

        def visit(syndrome, remaining, word):
            if cap and len(found) >= cap:
                return
            if not syndrome:
                found.add(word)
                return
            if syndrome.bit_count() > degree*remaining:
                return
            if remaining <= 2:
                tail = tails.get(syndrome)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    found.add(word | tail)
                return
            pending, options = syndrome, None
            while pending:
                bit = pending & -pending
                available = [j for j in incidence[bit.bit_length()-1]
                             if not bits[j] & (word | forbidden)]
                if not available:
                    return
                if options is None or len(available) < len(options):
                    options = available
                pending ^= bit
            options.sort(key=lambda j: -(cols[j] & syndrome).bit_count())
            for j in options:
                visit(syndrome ^ cols[j], remaining-1, word | bits[j])
                if cap and len(found) >= cap:
                    return

        visit(value, limit, 0)
        return tuple(sorted(found, key=lambda w: (w.bit_count(), w)))

    return decode


def _s48_low4(cols, stabilizers):
    """Exact logical existence through four, modulo actual stabilizers."""
    span = _dic_basis(stabilizers)
    seen = {0: 0}
    for j, value in enumerate(cols):
        word = 1 << j
        old = seen.get(value)
        if old is None:
            seen[value] = word
        elif _dic_outside(word ^ old, span):
            return True
    for j, value in enumerate(cols):
        for h in range(j):
            syndrome = value ^ cols[h]
            word = (1 << j) | (1 << h)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _dic_outside(word ^ old, span):
                return True
    return False


@lru_cache(maxsize=1)
def _s48_bank(G):
    """Pure fresh-label optimization for the new dicyclic shape bed."""
    from itertools import combinations
    from math import gcd

    assert G.q == 48 and A_SHAPE == (1,2) and B_SHAPE == (2,5)
    q, mask = G.q, (1 << G.q)-1
    za = [[[], []]]
    zb = [[[] for _ in range(5)] for _ in range(2)]
    atlas = {}
    for side in ("A", "B"):
        atoms = []
        for g in G.elements:
            A = [[[g], []]] if side == "A" else za
            B = ([[[g]]+[[] for _ in range(4)],
                  [[] for _ in range(5)]] if side == "B" else zb)
            hx, hz, _ = lifted_matrices(G, A, B)
            _, _, ac, bc = _component_data(hx, hz, q)
            atom = tuple((ac if side == "A" else bc)[:q])
            assert all(s.bit_count() == 1 for s in atom)
            atoms.append(atom)
        atlas[side] = tuple(atoms)

    def compile_component(side, grid):
        cols = [0]*(len(grid[0])*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for h, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+h] ^= value << (i*q)
        return tuple(cols)

    # Normal C3=<x^8>. Its coset indicators are physical supports.
    # Quotient syndromes below are extracted from verified columns only.
    fibers = tuple(
        tuple(G.idx((a+8*t, b)) for t in range(3))
        for b in range(2) for a in range(8))
    fiber_bits = tuple(sum(1 << h for h in fiber) for fiber in fibers)

    def fold(cols, height):
        projected, supports = [], []
        for block in range(len(cols)//q):
            for fiber in fibers:
                word = sum(1 << (block*q+h) for h in fiber)
                value = _s48_syndrome(word, cols)
                compact = 0
                for row in range(height//q):
                    part = (value >> (row*q)) & mask
                    for t, orbit in enumerate(fiber_bits):
                        overlap = part & orbit
                        assert overlap in (0, orbit)
                        if overlap:
                            compact |= 1 << (row*16+t)
                projected.append(compact)
                supports.append(word)
        return tuple(projected), tuple(supports)

    def norm_seeds(cols, height, local):
        projected, supports = fold(cols, height)
        result = []
        for coefficients in _s48_words(projected, local, 3):
            word = 0
            while coefficients:
                bit = coefficients & -coefficients
                word ^= supports[bit.bit_length()-1]
                coefficients ^= bit
            assert word and not _s48_syndrome(word, cols)
            result.append(word)
        return tuple(result)

    # Enumerate intrinsic normalized quartic quotient supports, not
    # group-automorphism images of one saved code.
    quotient_elements = [(a,b) for b in range(2) for a in range(8)]
    records = []
    for triple in combinations(quotient_elements[1:], 3):
        cell = (G.e,) + triple
        cols = compile_component("A", [[cell]])
        projected, _ = fold(cols, q)
        rank = gf2_rank(projected)
        if rank <= 11 and _short_floor(projected) >= 4:
            records.append((cell, projected, rank))
    library = {}
    library_rng = random.Random(48122501)
    library_rng.shuffle(records)
    counts = {}
    for i, first in enumerate(records):
        for second in records[:i]:
            joined = first[1] + second[1]
            rank = gf2_rank(joined)
            if not 5 <= rank <= 11 or _short_floor(joined) < 4:
                continue
            counts[rank] = counts.get(rank, 0)+1
            bucket = library.setdefault(rank, [])
            pair = (first[0], second[0])
            if len(bucket) < 96:
                bucket.append(pair)
            else:
                position = library_rng.randrange(counts[rank])
                if position < 96:
                    bucket[position] = pair

    def draw_a(local, preference, batch):
        if library:
            ranks = sorted(library,
                           key=lambda r: (abs(r-preference), r))
            rank = ranks[min(batch, len(ranks)-1)]
            prototype = local.choice(library[rank])
            return [[
                [(a+8*local.randrange(3), b) for a,b in cell]
                for cell in prototype]]
        # A fresh-label fallback still faces all exact rank/rate tests.
        # It never calls an archived or historical optimizer.
        result = []
        for _ in range(2):
            cell = [G.e]
            while len(cell) < 4:
                g = local.choice(G.elements)
                if g not in cell and all(
                        G.mul(G.inv(h), g) != (12,0) for h in cell):
                    cell.append(g)
            result.append(cell)
        return [result]

    def draw_b(local, counts_grid):
        grid = []
        for row in counts_grid:
            out = []
            for size in row:
                if not size:
                    out.append([])
                    continue
                selected = None
                for _ in range(80):
                    if size == 2:
                        base = local.choice(G.elements)
                        step = local.choice(
                            [d for d in range(1,24) if gcd(d,24) == 1])
                        cell = [base, G.mul(base, (step,0))]
                    else:
                        cell = local.sample(G.elements, size)
                    if any(G.mul(G.inv(g), h) == (12,0)
                           for i,g in enumerate(cell) for h in cell[:i]):
                        continue
                    if size >= 3:
                        relatives = [G.mul(G.inv(cell[0]), g)
                                     for g in cell[1:]]
                        if not any(G.mul(g,h) != G.mul(h,g)
                                   for i,g in enumerate(relatives)
                                   for h in relatives[:i]):
                            continue
                    selected = cell
                    break
                if selected is None:
                    selected = local.sample(G.elements, size)
                out.append(selected)
            grid.append(out)
        return grid

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        geometry, model = divmod(slot % 20, 4)
        sb = _S48_B_COUNTS[geometry]
        sa = ((4,4),)
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(BASE_SEED*100003+48122573+100019*slot)
        preference = (6,8,10,11)[model]
        winner = None

        def component_pool(side, batch):
            raw, seen = [], set()
            height = q if side == "A" else 2*q
            quota = (384 if side == "A" else 256)+128*batch
            for attempt in range(quota):
                grid = (draw_a(local, preference, batch) if side == "A"
                        else draw_b(local, sb))
                label = tuple(tuple(sorted(c))
                              for row in grid for c in row)
                if label in seen:
                    continue
                seen.add(label)
                cols = compile_component(side, grid)
                rank = gf2_rank(cols)
                if side == "A" and not 35 <= rank <= 42:
                    continue
                if side == "B" and rank != 2*q:
                    continue
                if _short_floor(cols) < 5:
                    continue
                seeds = norm_seeds(cols, height, local)
                words = _s48_words(cols, local, 3, seeds)
                if not words:
                    continue
                rows = _columns(cols, height)
                item = dict(grid=grid, cols=cols, words=words,
                            rank=rank, d=words[0].bit_count(),
                            cycles=_component_cycles(rows))
                raw.append(item)
                raw.sort(key=lambda x: (
                    min(x["d"],18), -x["cycles"],
                    -x["rank"] if side == "A" else x["rank"]),
                    reverse=True)
                del raw[10:]

            for item in raw:
                decode = _s48_decoder(item["cols"], height)
                seeds = list(item["words"])
                # Constituent regular actions are transitive in each
                # block. This anchor reduction is not used on full codes.
                for bound in (6,8):
                    for anchor in range(0, len(item["cols"]), q):
                        hits = decode(item["cols"][anchor], bound-1,
                                      1 << anchor)
                        if hits:
                            seeds.append(hits[0] | (1 << anchor))
                    if any(w.bit_count() <= bound for w in seeds):
                        break
                item["words"] = _s48_words(
                    item["cols"], local, 24, seeds)
                item["d"] = min(item["d"],
                                item["words"][0].bit_count())
            raw.sort(key=lambda x: (
                min(x["d"],18), -x["cycles"],
                -x["rank"] if side == "A" else x["rank"]),
                reverse=True)
            return raw[:4]

        def merit(item):
            d = min(item["a"]["d"], item["b"]["d"],
                    item["dx"], item["dz"], 18)
            return (item["k"]*d*d/item["n"], d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        for batch in range(3):
            ap, bp = component_pool("A", batch), component_pool("B", batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (n == 576 and 275*n <= 1000*k < 325*n):
                        continue
                    _, _, ac, bc = _component_data(hx, hz, q)
                    assert tuple(ac) == a["cols"] and tuple(bc) == b["cols"]
                    d = min(a["d"], b["d"], 18)
                    proposals.append((k*d*d, a, b, hx, hz, n, k))
            proposals.sort(key=lambda x: x[0], reverse=True)
            finalists = []
            for _, a, b, hx, hz, n, k in proposals:
                xc, zc = _columns(hz,n), _columns(hx,n)
                if _s48_low4(xc,hx) or _s48_low4(zc,hz):
                    continue
                fp = (n, _dic_check_signature(list(hx)+list(hz)))
                if fp in emitted:
                    continue
                xs, zs = _dic_basis(hx), _dic_basis(hz)
                dx = dz = n+1
                for word in b["words"]:
                    for j in range(2):
                        full = word << (j*5*q)
                        assert not _s48_syndrome(full, xc)
                        if _dic_outside(full, xs):
                            dx = min(dx, full.bit_count())
                for word in a["words"]:
                    for beta in range(5):
                        full = sum(((word >> (j*q)) & mask)
                                   << ((j*5+beta)*q) for j in range(2))
                        assert not _s48_syndrome(full, zc)
                        if _dic_outside(full, zs):
                            dz = min(dz, full.bit_count())
                item = dict(A=a["grid"], B=b["grid"], a=a, b=b,
                            hx=hx, hz=hz, xc=xc, zc=zc,
                            n=n, k=k, fp=fp,
                            dx=_dic_probe(xc,local,6,dx,hx),
                            dz=_dic_probe(zc,local,6,dz,hz))
                finalists.append(item)
                if len(finalists) == 3:
                    break
            # Equal independent deeper budgets for every finalist.
            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"],local,18,item["dx"],item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"],local,18,item["dz"],item["hz"])
                if min(item["dx"],item["dz"]) > item["n"]:
                    continue
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[0] > 43.102:
                break

        if winner is None:
            bank.append(None)
            continue
        hx, hz = winner["hx"], winner["hz"]
        assert css_ok(hx,hz)
        assert max(r.bit_count() for r in hx+hz) <= W_MAX
        assert 275*576 <= 1000*winner["k"] < 325*576
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (winner["A"],winner["B"])
                   for row in grid for c in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={side: tuple(tuple(tuple(c) for c in row)
                               for row in winner[side])
                   for side in ("A","B")},
            k=winner["k"], card=(geometry,model),
            screen=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh, rank-legal complete filling for the new shape."""
    item = _s48_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh target-rate Dic12 two-row-B design passed")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Every new-shape cell is selected jointly and returned without draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order binomial support profiles preserve the cubic Dic21 distance advantage; every distinct nonuniform-count parent slot read d2s=5, while all five measured parent codes used five trinomials.
# REFUTED: Component-kernel selection alone removes mixed-sector failures on Dic21; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Constituent-through-eight exclusion plus bounded singleton-sector-2 screening ensures five Dic21 readings >=12; current-parent measured slots 20 and 22 instead retained mixed-sector X readings nine and eleven, while the other three leaders read twelve.
# REFUTED: Different cubic labels and rotation-pair-order cards necessarily identify inequivalent codes; the measured current program had only 40 distinct codes among 60 slots, including iso_dup slots 2,9,11 and 17.
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
