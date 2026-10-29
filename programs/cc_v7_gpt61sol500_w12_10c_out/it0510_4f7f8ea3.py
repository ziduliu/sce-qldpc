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
CONCEPT = ("C19, order 19, A=(2,4), B=(3,6): balanced augmentation-zero constituents, with a longer three-row B component, should supply five fresh measured [570,138,14] readings and improve cell-0.250 fitness above 44.8. The corresponding score is approximately 47.45; this is a falsifiable measurement target, not a distance certificate.")
GROUP = ("L2 order repair inside the unchanged cyclic construction rule and one-coordinate encoding. Factor thirty makes order 29 inadmissible and permits q<=19, giving n=570. Since 2 has order eighteen modulo nineteen, the nontrivial part of F2[C19] is one field. Zero augmentation and full nontrivial component ranks motivate k=138, but complete verified ranks decide admission. The parent's five [580,132,14] readings motivate retaining quartic columns, not a claimed C19 distance advantage.")
SHAPES = ("Target cell 0.250 with unequal (2,4)x(3,6): n=570, kb=114 and guaranteed rate 0.200. Components have variables/checks 76/38 and 114/57. Require component ranks 36 and 54 and actual assembled k=138, rate 0.242105. Parent weight-fourteen X and Z minimum families occupied sector-two Arow/Brow blocks. The new wide components place tested embeddings in sector-one Acol/Bcol blocks; complete logical probes still inspect both sectors.")
FILLING = ("Every component row totals eight and every column totals four, so HX=HZ=12. Twenty cards cross three multiplier-inequivalent four-gap A policies and two Sidon-quartet A policies with four connected balanced B multigraph architectures: zero, one, two or three quartet loops. Pack distinct within-row differences and minimize cross-row collisions. Compile exclusively from verified unit lifts, retain ordinary/divided two-row and three-row cofactor witnesses, screen constituent dependencies through eight, and independently deepen all retained finalists. Exact k=138, complete kernel-four exclusion, syndrome-checked embeddings, exact stabilizer membership and unequal uncoloured Tanner fingerprints gate emission. Found readings remain upper witnesses, never distance certificates.")
FAMILY_NAME = "r025_C19_24x36_balancedquartic_three_row_cofactor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    # Same cyclic rule; order repaired for size factor thirty.
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
    els = [(a,) for a in range(19)]

    def red(g):
        return (g[0] % 19,)

    def mul(g, h):
        return ((g[0] + h[0]) % 19,)

    def inv(g):
        return ((-g[0]) % 19,)

    def idx(g):
        return g[0] % 19

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 6)               # (rb, cb)


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


def _c29_component_rows(hx, hz, q):
    """Extract both new-shape actions from the VERIFIED full lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((hx[(i*4)*q+r] >> (j*4*q)) & mask) << (j*q)
            for j in range(4))
        for i in range(2) for r in range(q)
    ]
    br = [
        hz[a*q+r] & ((1 << (4*q)) - 1)
        for a in range(2) for r in range(q)
    ]
    return ar, br


@lru_cache(maxsize=1)
def _c29_atlas(G):
    """Linear contributions compiled exclusively from scaffold unit lifts."""
    q = G.q
    mask = (1 << q) - 1
    zero = [[[] for _ in range(4)] for _ in range(2)]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(4)] for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)
    return atlas


def _c29_compiled(G, atlas, side, grid):
    cols = [0] * (4*G.q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                for h, contribution in enumerate(atlas[side][G.idx(g)]):
                    cols[j*G.q+h] ^= contribution << (i*G.q)
    return tuple(cols)


def _c29_syndrome(word, cols):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= cols[bit.bit_length()-1]
        word ^= bit
    return syndrome


def _c29_short_witness(cols, q):
    """Exact constituent existence test through eight after clean4.

    The regular constituent action is transitive inside each q-column
    block, so four block-zero anchors cover all constituent supports.
    This symmetry is NOT asserted for complete quantum matrices.
    """
    bits = [1 << i for i in range(len(cols))]
    incidence = [[] for _ in range(2*q)]
    tails = {0: 0}
    for i, value in enumerate(cols):
        assert value.bit_count() == 4 and value not in tails
        tails[value] = bits[i]
        remaining = value
        while remaining:
            bit = remaining & -remaining
            incidence[bit.bit_length()-1].append(i)
            remaining ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            assert syndrome not in tails
            tails[syndrome] = bits[i] | bits[j]

    def visit(value, remaining, word, forbidden):
        if not value:
            return word
        if value.bit_count() > 4*remaining:
            return None
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not (tail & (word | forbidden))):
                return word | tail
            return None

        checks = []
        residual = value
        while residual:
            bit = residual & -residual
            r = bit.bit_length()-1
            choices = [j for j in incidence[r]
                       if not (bits[j] & (word | forbidden))]
            if not choices:
                return None
            checks.append(choices)
            residual ^= bit
        choices = min(checks, key=len)
        choices.sort(key=lambda j: -(cols[j] & value).bit_count())
        for j in choices:
            hit = visit(value ^ cols[j], remaining-1,
                        word | bits[j], forbidden)
            if hit is not None:
                return hit
        return None

    for bound in (6, 8):
        for anchor in range(0, len(cols), q):
            hit = visit(cols[anchor], bound-1, 0, bits[anchor])
            if hit is not None:
                witness = hit | bits[anchor]
                assert _c29_syndrome(witness, cols) == 0
                return witness
    # Zero is an exclusion flag, not a found distance-nine word.
    return 0


def _c29_probe(cols, local, trials, seeds=(), stabilizers=None,
               translations=()):
    """Retain genuine witnesses, including logical directions separately.

    High readings remain bounded-search upper estimates. No-witness
    sentinels are never credited as a distance certificate.
    """
    n = len(cols)
    span = None if stabilizers is None else _dic_basis(stabilizers)
    generators = () if stabilizers is None else tuple(stabilizers)
    memory = sorted(
        {w for w in seeds if w and _c29_syndrome(w, cols) == 0},
        key=lambda w: (w.bit_count(), w))[:32]
    best = n+1

    def logical(word):
        return word and (span is None or _dic_outside(word, span))

    def observe(word):
        nonlocal best
        if logical(word):
            best = min(best, word.bit_count())

    for word in memory:
        observe(word)
    order = list(range(n))

    for trial in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                dependencies.append(word)
                observe(word)

        shifted = []
        for word in sorted(set(memory+dependencies),
                           key=int.bit_count)[:6]:
            for permutation in translations:
                moved, remaining = 0, word
                while remaining:
                    bit = remaining & -remaining
                    moved ^= permutation[bit.bit_length()-1]
                    remaining ^= bit
                if moved and _c29_syndrome(moved, cols) == 0:
                    shifted.append(moved)
                    observe(moved)

        all_words = set(memory+dependencies+shifted)
        donors = sorted(all_words, key=int.bit_count)[:40]
        starts = sorted(all_words, key=int.bit_count)[:24]
        if span is not None:
            starts += sorted(
                (w for w in all_words if logical(w)),
                key=int.bit_count)[:12]

        discoveries = set(memory+shifted)
        for start in starts:
            word = start
            for sweep in range(2):
                before = word
                for direction in donors:
                    candidate = word ^ direction
                    if candidate and candidate.bit_count() < word.bit_count():
                        word = candidate
                if word == before:
                    break
            observe(word)
            discoveries.add(word)

        short = sorted(discoveries, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        if generators:
            logical_seeds = sorted(
                (w for w in discoveries | all_words if logical(w)),
                key=int.bit_count)[:8]
            for start in logical_seeds:
                word = start
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in generators),
                        key=lambda x: x[0])[:12]
                    for _, row in nearby:
                        candidate = word ^ row
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)

        discoveries.update(dependencies)
        if span is None:
            memory = sorted(discoveries, key=int.bit_count)[:32]
        else:
            ordinary = sorted(discoveries, key=int.bit_count)[:16]
            logical_words = sorted(
                (w for w in discoveries if logical(w)),
                key=int.bit_count)[:16]
            memory = sorted(set(ordinary+logical_words),
                            key=lambda w: (w.bit_count(), w))
    assert all(w and _c29_syndrome(w, cols) == 0 for w in memory)
    return best, tuple(memory)


@lru_cache(maxsize=1)
def _c29_geometries():
    """Four-gap orbits under every C29 multiplier and orientation."""
    from itertools import combinations
    q = 29
    representatives = set()
    for ds in combinations(range(1, 15), 4):
        canonical = min(
            tuple(sorted(min((u*d) % q, q-(u*d) % q) for d in ds))
            for u in range(1, q))
        representatives.add(canonical)

    def triangles(ds):
        return sum(
            any((a+sign*b) % q in (c, q-c) for sign in (-1, 1))
            for a, b, c in combinations(ds, 3))
    result = tuple(sorted(representatives,
                          key=lambda ds: (triangles(ds), ds)))
    assert len(result) >= 12
    return result


def _c29_draw(local, G, top, intersection):
    """Pack a prescribed gap geometry, not a translated parental filling."""
    q = G.q
    top_set = set(top)
    outside = [d for d in range(1, 15) if d not in top_set]
    for restart in range(96):
        lower = (local.sample(list(top), intersection)
                 + local.sample(outside, 4-intersection))
        local.shuffle(lower)
        upper = [d*local.choice((-1, 1)) % q for d in top]
        lower = [d*local.choice((-1, 1)) % q for d in lower]
        if any(upper[j] in (lower[j], (-lower[j]) % q)
               for j in range(4)):
            continue
        used, offsets = set(), [None]*4
        order = list(range(4))
        local.shuffle(order)
        for j in order:
            choices = []
            d, e = upper[j], lower[j]
            for a in range(q):
                cross = {(a-x+y) % q
                         for x in (0, d) for y in (0, e)}
                if len(cross) == 4 and not (cross & used):
                    choices.append((a, cross))
            if not choices:
                break
            a, cross = local.choice(choices)
            offsets[j] = a
            used.update(cross)
        if all(a is not None for a in offsets):
            return [
                [[G.e, (upper[j],)] for j in range(4)],
                [[(offsets[j],), ((offsets[j]+lower[j]) % q,)]
                 for j in range(4)],
            ]
    return None


def _c29_component_pool(local, G, atlas, side, top, intersection,
                        translations, quota):
    """Fresh labels, exact short screening, and equal deeper budgets."""
    pool, seen = [], set()
    for attempt in range(quota):
        grid = _c29_draw(local, G, top, intersection)
        if grid is None:
            continue
        signature = tuple(tuple(sorted(c)) for row in grid for c in row)
        if signature in seen:
            continue
        seen.add(signature)
        cols = _c29_compiled(G, atlas, side, grid)
        assert all(c.bit_count() == 4 for c in cols)
        if gf2_rank(cols) != 2*G.q-2 or _short_floor(cols) < 5:
            continue
        reading, words = _c29_probe(
            cols, local, 2, translations=translations)
        if reading > len(cols):
            continue
        item = dict(grid=grid, cols=cols, d=reading, words=words,
                    attempt=attempt, clean8=False)
        pool.append(item)
        pool.sort(key=lambda x: (x["d"], -x["attempt"]), reverse=True)
        del pool[8:]

    for item in pool:
        witness = _c29_short_witness(item["cols"], G.q)
        item["clean8"] = not witness
        seeds = list(item["words"])
        if witness:
            seeds.append(witness)
            item["d"] = min(item["d"], witness.bit_count())
        reading, words = _c29_probe(
            item["cols"], local, 16, seeds,
            translations=translations)
        item["d"] = min(item["d"], reading)
        item["words"] = words
    pool.sort(key=lambda x: (
        x["d"], int(x["clean8"]), -x["attempt"]), reverse=True)
    leaders = pool[:3]
    for item in leaders:
        reading, words = _c29_probe(
            item["cols"], local, 48, item["words"],
            translations=translations)
        item["d"] = min(item["d"], reading)
        item["words"] = words
    leaders.sort(key=lambda x: (
        x["d"], int(x["clean8"]), -x["attempt"]), reverse=True)
    return leaders


def _c29_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; inequality certifies nonisomorphism."""
    from collections import Counter
    rows = list(hx)+list(hz)
    adjacency = [0]*len(rows)
    overlaps = [dict() for _ in rows]
    incidents = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        remaining = row
        while remaining:
            bit = remaining & -remaining
            incidents[bit.bit_length()-1].append(i)
            remaining ^= bit
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = overlap

    keys = []
    for i, neighbors in enumerate(overlaps):
        profile = Counter(
            (rows[j].bit_count(), overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        keys.append((rows[i].bit_count(), tuple(sorted(profile.items()))))
    history = []
    for _ in range(2):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: i for i, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], overlap)
                for j, overlap in neighbors.items()).items())))
            for i, neighbors in enumerate(overlaps)
        ]
    qubits = Counter(
        tuple(sorted(rows[i].bit_count() for i in neighbors))
        for neighbors in incidents)
    return n, tuple(history), tuple(sorted(qubits.items()))


@lru_cache(maxsize=1)
def _c29_design_bank(G):
    """Twenty deliberate fresh policies; never replay an emitted design."""
    assert G.q == 29 and A_SHAPE == B_SHAPE == (2, 4)
    q = G.q
    mask = (1 << q)-1
    counts = [[2]*4 for _ in range(2)]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)
    atlas = _c29_atlas(G)
    geometries = _c29_geometries()

    # These permutations move witnessed supports, not cell labels.
    # Every moved support is syndrome-checked before entering memory.
    translations = []
    for t in (1, 7, 13):
        permutation = [G.idx(G.mul(g, (t,))) for g in G.elements]
        translations.append(tuple(
            1 << (j*q+permutation[h])
            for j in range(4) for h in range(q)))
    translations = tuple(translations)
    emitted, bank = set(), []

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item["dx"], item["dz"])
        return (item["k"]*d*d/item["n"], d,
                item["a"]["d"]+item["b"]["d"])

    for slot in range(N_SLOTS):
        geometry, intersection = divmod(slot % 20, 4)
        lane = slot // 20
        atop = geometries[(geometry+5*lane) % len(geometries)]
        btop = geometries[(geometry+7+7*lane) % len(geometries)]
        local = random.Random(
            BASE_SEED*100003 + 290250917 + 104729*slot)
        winner = None

        for batch in range(2):
            ap = _c29_component_pool(
                local, G, atlas, "A", atop, intersection,
                translations, 96+48*batch)
            bp = _c29_component_pool(
                local, G, atlas, "B", btop, intersection,
                translations, 96+48*batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 580 or k != 132:
                        continue
                    ar, br = _c29_component_rows(hx, hz, q)
                    assert tuple(_columns(ar, 4*q)) == a["cols"]
                    assert tuple(_columns(br, 4*q)) == b["cols"]
                    d = min(a["d"], b["d"])
                    proposals.append((
                        (d, a["d"]+b["d"]), a, b, hx, hz, n, k))
            proposals.sort(key=lambda x: x[0], reverse=True)

            finalists = []
            for _, a, b, hx, hz, n, k in proposals:
                xc, zc = _columns(hz, n), _columns(hx, n)
                # Conservative complete-kernel exclusion, including both
                # LP sectors. A small stabilizer is rejected as well.
                if _short_floor(xc) < 5 or _short_floor(zc) < 5:
                    continue
                fingerprint = _c29_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue

                xseeds = [
                    word << (j*4*q)
                    for word in b["words"][:16] for j in range(4)]
                zseeds = [
                    sum(((word >> (j*q)) & mask)
                        << ((j*4+beta)*q) for j in range(4))
                    for word in a["words"][:16] for beta in range(4)]
                xseeds = [w for w in xseeds
                          if _c29_syndrome(w, xc) == 0]
                zseeds = [w for w in zseeds
                          if _c29_syndrome(w, zc) == 0]
                dx, xwords = _c29_probe(
                    xc, local, 6, xseeds, hx)
                dz, zwords = _c29_probe(
                    zc, local, 6, zseeds, hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=a["grid"], B=b["grid"], a=a, b=b,
                    hx=hx, hz=hz, xc=xc, zc=zc, n=n, k=k,
                    dx=dx, dz=dz, xwords=xwords, zwords=zwords,
                    fingerprint=fingerprint))
                if len(finalists) == 2:
                    break

            # Identical independent budgets for every assembled finalist;
            # previously discovered lighter logicals remain binding.
            for item in finalists:
                dx, _ = _c29_probe(
                    item["xc"], local, 14, item["xwords"], item["hx"])
                dz, _ = _c29_probe(
                    item["zc"], local, 14, item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 15:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["n"] == 580 and winner["k"] == 132
        assert 225*winner["n"] <= 1000*winner["k"] < 275*winner["n"]
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 2
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(winner["fingerprint"])
        frozen = {
            side: tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for side, grid in (("A", A), ("B", B))}
        bank.append(dict(cells=frozen, k=132, screen=merit(winner),
                         policy=(geometry, intersection, lane)))
    return tuple(bank)


def _r19_parts(hx, hz, q):
    """Extract arbitrary-shaped constituents from the VERIFIED full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = [
        sum(((hx[(i*cb)*q+t] >> (j*cb*q)) & mask) << (j*q)
            for j in range(ca))
        for i in range(ra) for t in range(q)]
    br = [
        hz[a*q+t] & ((1 << (cb*q))-1)
        for a in range(rb) for t in range(q)]
    return ar, br


def _r19_cofactors(G, side, grid, cols):
    """Two-/three-row syzygies; all candidate supports are syndrome-tested."""
    from itertools import combinations, permutations
    q, rows, width = G.q, len(grid), len(grid[0])
    entries = [
        [{G.idx(G.inv(g) if side == "B" else g) for g in cell}
         for cell in row] for row in grid]

    def divide(cell):
        polynomial = sum(1 << a for a in cell)
        quotient = 0
        while polynomial:
            p = polynomial.bit_length()-1
            assert p > 0
            quotient ^= 1 << (p-1)
            polynomial ^= 3 << (p-1)
        return {a for a in range(q) if quotient & (1 << a)}

    def multiply(left, right):
        out = set()
        for a in left:
            for b in right:
                c = (a+b) % q
                if c in out:
                    out.remove(c)
                else:
                    out.add(c)
        return out

    divided = [[divide(cell) for cell in row] for row in entries]
    found = set()
    for matrix in (entries, divided):
        minors = {}
        for chosen in combinations(range(width), rows):
            determinant = set()
            for order in permutations(chosen):
                term = {0}
                for i, j in enumerate(order):
                    term = multiply(term, matrix[i][j])
                    if not term:
                        break
                determinant ^= term
            minors[chosen] = determinant
        for support in combinations(range(width), rows+1):
            word = 0
            for omitted in support:
                chosen = tuple(j for j in support if j != omitted)
                for a in minors[chosen]:
                    word ^= 1 << (omitted*q+a)
            if word and _c29_syndrome(word, cols) == 0:
                found.add(word)
    # Explicit constant-fiber witnesses prevent rewarding missed fiber words.
    mask = (1 << q)-1
    for j in range(width):
        word = mask << (j*q)
        if _c29_syndrome(word, cols) == 0:
            found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w)))


def _r19_short(cols, q, height):
    """Exact constituent search through eight after clean4, not a certificate."""
    bits = [1 << j for j in range(len(cols))]
    incidence = [0]*height
    tails = {0: 0}
    for j, value in enumerate(cols):
        assert value.bit_count() == 4 and value not in tails
        tails[value] = bits[j]
        pending = value
        while pending:
            bit = pending & -pending
            incidence[bit.bit_length()-1] |= bits[j]
            pending ^= bit
    for j, value in enumerate(cols):
        for h in range(j):
            syndrome = value ^ cols[h]
            assert syndrome not in tails
            tails[syndrome] = bits[j] | bits[h]

    def solve(value, remaining, banned):
        if not value:
            return 0
        if remaining <= 0 or value.bit_count() > 4*remaining:
            return None
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not (tail & banned)):
                return tail
            return None
        pending, choices, least = value, 0, len(cols)+1
        while pending:
            bit = pending & -pending
            options = incidence[bit.bit_length()-1] & ~banned
            amount = options.bit_count()
            if not amount:
                return None
            if amount < least:
                choices, least = options, amount
            pending ^= bit
        order = []
        while choices:
            bit = choices & -choices
            order.append(bit.bit_length()-1)
            choices ^= bit
        order.sort(key=lambda j: (-(cols[j] & value).bit_count(), j))
        excluded = banned
        for j in order:
            excluded |= bits[j]
            tail = solve(value ^ cols[j], remaining-1, excluded)
            if tail is not None:
                return bits[j] | tail
        return None

    # Regular constituent translation is transitive inside each block.
    # This reduction is not used for complete quantum matrices.
    for bound in (6, 8):
        for anchor in range(0, len(cols), q):
            tail = solve(cols[anchor], bound-1, bits[anchor])
            if tail is not None:
                word = bits[anchor] | tail
                assert _c29_syndrome(word, cols) == 0
                return word
    return 0


@lru_cache(maxsize=1)
def _r19_bank(G):
    """Twenty fresh structural tasks on the repaired cyclic rectangular bed."""
    from itertools import combinations
    assert G.q == 19 and A_SHAPE == (2, 4) and B_SHAPE == (3, 6)
    q, mask = G.q, (1 << G.q)-1
    assert next(t for t in range(1, 19) if pow(2, t, 19) == 1) == 18
    shapes = {"A": A_SHAPE, "B": B_SHAPE}
    zeros = {
        side: [[[] for _ in range(c)] for _ in range(r)]
        for side, (r, c) in shapes.items()}

    # F2-linear superposition atoms come only from scaffold unit-cell lifts.
    atlas = {}
    for side, (rows, width) in shapes.items():
        atoms = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in zeros["A"]]
            B = [[list(c) for c in row] for row in zeros["B"]]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 570
            ar, br = _r19_parts(hx, hz, q)
            source = ar if side == "A" else br
            atom = tuple(_columns(source, width*q)[:q])
            assert all(v.bit_count() == 1 and v.bit_length() <= q
                       for v in atom)
            atoms.append(atom)
        atlas[side] = tuple(atoms)

    def compile_columns(side, grid):
        cols = [0]*(len(grid[0])*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for h, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+h] ^= value << (i*q)
        return tuple(cols)

    def gap(a, b):
        d = (a-b) % q
        return min(d, q-d)

    gap_orbits = set()
    for ds in combinations(range(1, 10), 4):
        gap_orbits.add(min(
            tuple(sorted(gap(u*d, 0) for d in ds))
            for u in range(1, q)))
    gap_cards = tuple(sorted(gap_orbits))
    quartets = set()
    for a, b in combinations(range(2, q), 2):
        cell = (0, 1, a, b)
        if len({gap(x, y) for x, y in combinations(cell, 2)}) != 6:
            continue
        quartets.add(min(
            tuple(sorted((pow((y-x) % q, -1, q)*(z-x)) % q
                         for z in cell))
            for x in cell for y in cell if x != y))
    quartet_cards = tuple(sorted(quartets))
    assert len(gap_cards) >= 3 and len(quartet_cards) >= 2

    bplans = []
    for loops in ((0,0,0), (1,0,0), (1,1,0), (1,1,1)):
        columns = []
        for i, amount in enumerate(loops):
            for _ in range(amount):
                columns.append(tuple(4 if h == i else 0 for h in range(3)))
        for i, j, other in ((0,1,2), (0,2,1), (1,2,0)):
            amount = 2-loops[i]-loops[j]+loops[other]
            for _ in range(amount):
                columns.append(tuple(2 if h in (i,j) else 0
                                     for h in range(3)))
        counts = [[column[i] for column in columns] for i in range(3)]
        assert len(columns) == 6
        assert all(sum(row) == 8 for row in counts)
        assert all(sum(column) == 4 for column in columns)
        bplans.append(counts)

    translations = {
        side: tuple(tuple(
            1 << (j*q+(h+t) % q)
            for j in range(width) for h in range(q))
            for t in (1, 5, 9))
        for side, (_, width) in shapes.items()}
    bank, emitted = [], set()

    for slot in range(N_SLOTS):
        policy, architecture = divmod(slot % 20, 4)
        lane = slot//20
        local = random.Random(BASE_SEED*100003 + 192436251 + 104729*slot)
        sa = ([[2]*4 for _ in range(2)] if policy < 3 else
              [[4,0,2,2], [0,4,2,2]])
        sb = bplans[architecture]
        counts_by_side = {"A": sa, "B": sb}
        assert _grid_weights(sa, sb) == 12 and _no_dead_lines(sa, sb)

        def draw(side):
            counts = counts_by_side[side]
            rows, width = shapes[side]
            grid = [[[] for _ in range(width)] for _ in range(rows)]
            used = [set() for _ in range(rows)]
            fixed = None
            if side == "A" and policy < 3:
                base = gap_cards[(policy+3*lane) % len(gap_cards)]
                unit = local.randrange(1, q)
                fixed = [gap(unit*d, 0) for d in base]

            # Fill quartet loops first, reserving their six unsigned gaps.
            for i, row in enumerate(counts):
                for j, count in enumerate(row):
                    if count != 4:
                        continue
                    geometry = quartet_cards[
                        (policy-3+i+j+lane) % len(quartet_cards)]
                    unit, offset = local.randrange(1, q), local.randrange(q)
                    cell = [(unit*a+offset) % q for a in geometry]
                    used[i].update(gap(a,b) for a,b in combinations(cell,2))
                    grid[i][j] = [(a,) for a in cell]

            order = [j for j in range(width)
                     if any(counts[i][j] == 2 for i in range(rows))]
            local.shuffle(order)
            crossings = {}
            for j in order:
                occupied = [i for i in range(rows) if counts[i][j] == 2]
                previous_gap = None
                for i in occupied:
                    legal = [d for d in range(1, 10)
                             if d not in used[i] and d != previous_gap]
                    if fixed is not None and i == 0:
                        d = fixed[j]
                        if d not in legal:
                            return None
                    elif legal:
                        d = local.choice(legal)
                    else:
                        return None
                    used[i].add(d)
                    previous_gap = d
                    delta = d*local.choice((-1,1)) % q
                    grid[i][j] = [(0,), (delta,)]

                if len(occupied) == 2:
                    i, h = occupied
                    old = crossings.setdefault((i,h), set())
                    candidates = []
                    for offset in range(q):
                        lower = [((g[0]+offset) % q,) for g in grid[h][j]]
                        cross = {(a[0]-b[0]) % q
                                 for a in grid[i][j] for b in lower}
                        cost = (4-len(cross), len(cross & old))
                        candidates.append((cost, lower, cross))
                    best = min(item[0] for item in candidates)
                    _, lower, cross = local.choice(
                        [item for item in candidates if item[0] == best])
                    grid[h][j] = lower
                    old.update(cross)
            return grid

        def pool(side, batch):
            rows, width = shapes[side]
            height = rows*q
            raw, seen = [], set()

            def key(item):
                d = item["d"]
                load = sum(w.bit_count() <= d+1 for w in item["words"])
                return (d, int(item["clean8"]), -load,
                        -item["cycles"], -item["serial"])

            quota = (384 if side == "A" else 256)+128*batch+16*lane
            for attempt in range(quota):
                grid = draw(side)
                if grid is None:
                    continue
                labels = tuple(tuple(sorted(c)) for row in grid for c in row)
                if labels in seen:
                    continue
                seen.add(labels)
                cols = compile_columns(side, grid)
                assert all(v.bit_count() == 4 for v in cols)
                if gf2_rank(cols) != rows*(q-1) or _short_floor(cols) < 5:
                    continue
                cycles = _component_cycles(_columns(cols, height))
                seeds = (_r19_cofactors(G, side, grid, cols)
                         if side == "A" else
                         tuple(mask << (j*q) for j in range(width)))
                d, words = _c29_probe(
                    cols, local, 2, seeds, translations=translations[side])
                if d > len(cols):
                    continue
                raw.append(dict(
                    grid=grid, cols=cols, d=d, words=words,
                    cycles=cycles, clean8=False, serial=attempt))
                raw.sort(key=key, reverse=True)
                del raw[8:]

            for item in raw:
                hit = _r19_short(item["cols"], q, height)
                item["clean8"] = not hit
                seeds = item["words"] + _r19_cofactors(
                    G, side, item["grid"], item["cols"])
                if hit:
                    seeds += (hit,)
                    item["d"] = min(item["d"], hit.bit_count())
                d, words = _c29_probe(
                    item["cols"], local, 16, seeds,
                    translations=translations[side])
                item["d"], item["words"] = min(item["d"], d), words
            raw.sort(key=key, reverse=True)
            leaders = raw[:3]
            for item in leaders:
                d, words = _c29_probe(
                    item["cols"], local, 48, item["words"],
                    translations=translations[side])
                item["d"], item["words"] = min(item["d"], d), words
            return sorted(leaders, key=key, reverse=True)

        def merit(item):
            d = min(item["a"]["d"], item["b"]["d"],
                    item.get("dx", 571), item.get("dz", 571), 19)
            return (item["k"]*d*d/570, d,
                    item["a"]["d"]+item["b"]["d"],
                    -item["a"]["cycles"]-item["b"]["cycles"])

        winner = None
        for batch in range(2):
            ap, bp = pool("A", batch), pool("B", batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 570 or k != 138:
                        continue
                    ar, br = _r19_parts(hx, hz, q)
                    assert tuple(_columns(ar, 4*q)) == a["cols"]
                    assert tuple(_columns(br, 6*q)) == b["cols"]
                    proposals.append(dict(
                        A=a["grid"], B=b["grid"], a=a, b=b,
                        hx=hx, hz=hz, n=n, k=k))
            proposals.sort(key=merit, reverse=True)
            finalists, local_signatures = [], set()
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xc, zc = _columns(hz, n), _columns(hx, n)
                if _short_floor(xc) < 5 or _short_floor(zc) < 5:
                    continue
                signature = _c29_fingerprint(hx, hz, n)
                if signature in emitted or signature in local_signatures:
                    continue
                local_signatures.add(signature)
                xseeds = [
                    word << (j*6*q)
                    for word in item["b"]["words"][:16] for j in range(4)]
                zseeds = [
                    sum(((word >> (j*q)) & mask)
                        << ((j*6+beta)*q) for j in range(4))
                    for word in item["a"]["words"][:16]
                    for beta in range(6)]
                assert all(_c29_syndrome(w, xc) == 0 for w in xseeds)
                assert all(_c29_syndrome(w, zc) == 0 for w in zseeds)
                dx, wx = _c29_probe(xc, local, 4, xseeds, hx)
                dz, wz = _c29_probe(zc, local, 4, zseeds, hz)
                if dx > n or dz > n:
                    continue
                item.update(
                    xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                    signature=signature)
                finalists.append(item)
                if len(finalists) == 3:
                    break
            # Equal independent full-code budgets; previous minima bind.
            for item in finalists:
                dx, _ = _c29_probe(
                    item["xc"], local, 20, item["wx"], item["hx"])
                dz, _ = _c29_probe(
                    item["zc"], local, 20, item["wz"], item["hz"])
                item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 14:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert lifted_k(G, A, B) == winner["k"] == 138
        assert winner["n"] == 570
        assert 225*570 <= 1000*138 < 275*570
        assert max(r.bit_count() for r in winner["hx"]+winner["hz"]) == 12
        assert [[len(c) for c in row] for row in A] == sa
        assert [[len(c) for c in row] for row in B] == sb
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A,B) for row in grid for c in row)
        emitted.add(winner["signature"])
        frozen = {
            side: tuple(tuple(tuple(c) for c in row) for row in grid)
            for side, grid in (("A",A), ("B",B))}
        bank.append(dict(
            cells=frozen, k=138, screen=merit(winner),
            policy=(policy, architecture, lane)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this card's fresh rank-gated two-/three-row cyclic design."""
    selected = _r19_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh C19 k=138 filling passed this structural card")
    return {"idx": idx, "cells": selected["cells"],
            "k": selected["k"], "screen": selected["screen"],
            "policy": selected["policy"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell at every new rectangular index."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order binomial support profiles preserve the cubic Dic21 distance advantage; every distinct nonuniform-count parent slot read d2s=5, while all five measured parent codes used five trinomials.
# REFUTED: Component-kernel selection alone removes mixed-sector failures on Dic21; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Exact cubic constituent-through-eight and singleton-sector-2 screening ensure five Dic21 readings >=12; the measured leaders instead read 9,11,12,12,12, with the two lighter X witnesses touching both sectors.
# REFUTED: Four-cycle-free all-binomial C29 constituents and through-eight screening remove the sector-one weight-fourteen bottleneck; all five measured current-parent codes remained [580,132,14], with both minimum families entirely in sector one.
# REFUTED: Tall C29 quartet/zero constituents, Sidon packing and divided-cofactor screening supply five readings >=15; all five measured parent leaders instead remained [580,132,14], with both minimum families entirely in sector two.
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
