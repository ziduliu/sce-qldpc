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
CONCEPT = ("Dic21, order 84, A=(1,2), B=(1,3): preferentially exclude subgroup-invariant constituent words through physical weight ten, then screen mixed-sector logicals, to raise five-code fitness above the parent's measured 30.7648 in cell 0.300.")
GROUP = ("Frozen L1 group, unchanged: this exact Dic21 bed produced a measured distance-12 code at k=168, while four other measured leaders retained distance 9 or 10. Its subgroups of orders 2,3,4,7 supply concrete invariant-subspace tests; their exclusion is not a quantum-distance certificate.")
SHAPES = ("Frozen (1,2)x(1,3): n=588, kb=168, guaranteed rate 2/7, constituent variable/check counts 168/84 and 252/84, and size-law q<=85. Parent weight-ten X minima occupied sector-one Acol/Bcol blocks, often emphasizing Acol1; slot 21's weight-nine X minimum also touched Arow0/Brow0. Require full constituent row ranks and verify k=168 rather than rewarding rank-deficit proxy inflation.")
FILLING = ("Five mixed trinomials retain HX/HZ weights 9/12. Independent pair-order cards pack differences, then verified constituent syndromes are folded along commuting subgroup translations of orders 2,3,4,7. An anchored exact central-fold test excludes physical weights <=10; other folds exclude their invariant words through the same threshold. Witness-capped fallbacks remain eligible. Existing deeper constituent and stabilizer-tested full-code probes screen survivors. Fresh deterministic seeds and permutation-invariant Tanner fingerprints prevent emitted orbit copies.")
FAMILY_NAME = "r030_dic21_12x13_cubic_norm2374_rankfull_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Dic21: x^42=1, y^2=x^21, y*x*y^-1=x^-1.
    if q != 84:
        raise ValueError("Dic21 has order 84")
    els = [(a, b) for b in range(2) for a in range(42)]

    def red(g):
        a, b = g
        return ((a + 21 * (b // 2)) % 42, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 21*b*d) % 42,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 42, 0) if b == 0 else ((a + 21) % 42, 1)

    def idx(g):
        a, b = red(g)
        return 42*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def _norm_optimize(rng, G, idx):
    """Optimize one prescribed card using verified invariant subspaces.

    Fold exclusions concern classical constituent kernels. Full-code
    logical probes still test stabilizer membership and both sectors.
    """
    from math import gcd
    from itertools import combinations_with_replacement
    a_cards = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3))
    b_cards = ((1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3))
    cards = [(a, b) for a in a_cards for b in b_cards]
    # Additional slots get new architectures, not repeated twenty-card
    # batches or transported copies of a selected labeling.
    cards.extend(
        (a, b)
        for a in combinations_with_replacement((1, 2, 3, 6), 2)
        for b in combinations_with_replacement((1, 2, 3, 6), 3)
        if (a, b) not in cards
    )
    card = idx % len(cards)
    ad, bd = cards[card]
    # Rotation-pair orders 42,21,14,7 respectively.
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

    def probe(cols, trials, start=None, stabilizers=None):
        """Actual kernel witnesses with information-set recombination.

        For full-code searches, a witness lowers the reading only when it
        lies outside the matching stabilizer row space. Kernel descent
        may use stabilizers, but a stabilizer never becomes a logical.
        """
        best = len(cols) + 1 if start is None else start
        order = list(range(len(cols)))
        memory = []

        def consider(word):
            nonlocal best
            weight = word.bit_count()
            if not word or weight >= best:
                return
            if stabilizers is None or outside(word, stabilizers):
                best = weight

        for trial in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = []
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

            words.sort(key=int.bit_count)
            discoveries = set(words[:40])
            discoveries.update(memory)
            cutoff = min(best + 4, len(cols))
            for i, word in enumerate(words):
                for other in words[:i]:
                    combined = word ^ other
                    if combined and combined.bit_count() <= cutoff:
                        consider(combined)
                        discoveries.add(combined)

            # Recombine short words from different information sets.
            short = sorted(discoveries, key=int.bit_count)[:48]
            for i, word in enumerate(short):
                for other in short[:i]:
                    combined = word ^ other
                    if combined and combined.bit_count() <= cutoff:
                        consider(combined)
                        discoveries.add(combined)

            reducers = words + memory
            seeds = sorted(discoveries, key=int.bit_count)[:6]
            for seed in seeds:
                word = seed
                for step in range(3):
                    chosen, weight = word, word.bit_count()
                    for reducer in reducers:
                        combined = word ^ reducer
                        if combined and combined.bit_count() < weight:
                            chosen, weight = combined, combined.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                    consider(word)
                    discoveries.add(word)
            memory = sorted(discoveries, key=int.bit_count)[:48]
            if best <= 4:
                break
        return best

    # These partitions describe commuting translations, not a separately
    # implemented lift. All parity syndromes below come from the scaffold.
    partitions = {}
    for side in ("A", "B"):
        for order, generator in (
                (2, (21, 0)), (3, (14, 0)),
                (4, (0, 1)), (7, (6, 0))):
            covered, orbits = set(), []
            for g in G.elements:
                if G.idx(g) in covered:
                    continue
                h, orbit = g, []
                for _ in range(order):
                    orbit.append(G.idx(h))
                    h = (G.mul(h, generator) if side == "A"
                         else G.mul(generator, h))
                assert G._key(h) == G._key(g)
                assert len(set(orbit)) == order
                covered.update(orbit)
                orbits.append(tuple(orbit))
            assert len(covered) == G.q
            partitions[side, order] = tuple(orbits)

    def fold(cols, side, order):
        """A folded support of weight t expands to physical weight order*t."""
        orbits = partitions[side, order]
        masks = [sum(1 << h for h in orbit) for orbit in orbits]
        out = []
        for block in range(len(cols) // G.q):
            for orbit in orbits:
                syndrome = 0
                for h in orbit:
                    syndrome ^= cols[block * G.q + h]
                compact = 0
                for r, mask in enumerate(masks):
                    restricted = syndrome & mask
                    assert restricted == 0 or restricted == mask
                    if restricted:
                        compact |= 1 << r
                out.append(compact)
        return out

    def central_witness5(cols):
        """Exactly exclude central-fold kernel weights <=5, or find one.

        The central order-two quotient has a transitive regular action
        on each block. Translate an occupied coordinate to that block's
        anchor. Every support of size five then has a 3+2 partition with
        the anchor in the triple. Smaller supports use singleton/pair
        collisions or the same anchored test.
        """
        seen = {0: 0}
        bits = [1 << i for i in range(len(cols))]
        for i, syndrome in enumerate(cols):
            word = bits[i]
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
        for i, syndrome in enumerate(cols):
            for j in range(i):
                word = bits[i] | bits[j]
                s = syndrome ^ cols[j]
                old = seen.get(s)
                if old is not None:
                    return (old ^ word).bit_count()
                seen[s] = word

        block_width = G.q // 2
        for anchor in range(0, len(cols), block_width):
            for i in range(len(cols)):
                if i == anchor:
                    continue
                s = cols[anchor] ^ cols[i]
                word = bits[anchor] | bits[i]
                for j in range(i):
                    if j == anchor:
                        continue
                    support = word | bits[j]
                    old = seen.get(s ^ cols[j])
                    if old is not None and old != support:
                        return (old ^ support).bit_count()
        return None

    def norm_screen(cols, side):
        """Return an actual invariant witness cap, or an unscored sentinel.

        Clean means exact exclusion through physical weight ten only
        within the four tested invariant subspaces.
        """
        upper = len(cols) + 1
        central = central_witness5(fold(cols, side, 2))
        if central is not None:
            upper = min(upper, 2 * central)
        for order in (3, 4, 7):
            floor = _short_floor(fold(cols, side, order))
            if floor < 5:
                # Here the short-floor value is an actual kernel weight.
                upper = min(upper, order * floor)
        return upper, upper > 10

    def score(candidate):
        d = min(candidate["da"], candidate["db"], candidate["full"])
        return (candidate["k"] * d * d / candidate["n"], d,
                sum(candidate["norm_clean"]),
                candidate["da"] + candidate["db"],
                -candidate["six"][1], -sum(candidate["six"]))

    # Exact restricted-kernel exclusions precede heuristic cycle ranking.
    # Full constituent row ranks imply the rank-count quantum dimension;
    # nevertheless the complete quantum ranks are checked below.
    raw = []
    for attempt in range(192):
        A, B = packed(ad), packed(bd)
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br, ac, bc = _component_data(hx, hz, G.q)
        if gf2_rank(ar) != G.q or gf2_rank(br) != G.q:
            continue
        if _component_cycles(ar) or _component_cycles(br):
            continue
        na, clean_a = norm_screen(ac, "A")
        nb, clean_b = norm_screen(bc, "B")
        ta, tb = six_cycles(ac), six_cycles(bc)
        structure = (
            not (clean_a and clean_b), not clean_b, not clean_a,
            -min(na, nb, 12), 2 * tb + ta, max(ta, tb), tb, ta,
        )
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
            "ac": ac, "bc": bc, "six": (ta, tb),
            "norm_caps": (na, nb), "norm_clean": (clean_a, clean_b),
            "structure": structure, "full": n + 1,
        }
        raw.append(candidate)
        raw.sort(key=lambda c: c["structure"])
        del raw[8:]

    pool = []
    for candidate in raw:
        hx, hz, n = candidate["hx"], candidate["hz"], candidate["n"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 168 or not 275 * n <= 1000 * k < 325 * n:
            continue
        candidate["k"] = k
        if min(_short_floor(candidate["ac"]),
               _short_floor(candidate["bc"])) < 5:
            continue
        # Preserve actual invariant witnesses in every subsequent reading.
        # A clean-screen sentinel is never itself counted as a witness.
        candidate["da"] = probe(
            candidate["ac"], 10, candidate["norm_caps"][0])
        candidate["db"] = probe(
            candidate["bc"], 14, candidate["norm_caps"][1])
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
            candidate["ac"], 32, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 48, candidate["db"])
        candidate["xs"] = row_basis(candidate["hx"])
        candidate["zs"] = row_basis(candidate["hz"])
        candidate["xc"] = _columns(candidate["hz"], candidate["n"])
        candidate["zc"] = _columns(candidate["hx"], candidate["n"])
        # Include both LP sectors. Low-weight stabilizers are explicitly
        # removed from the logical reading.
        upper = min(candidate["da"], candidate["db"])
        dx = probe(candidate["xc"], 14, upper, candidate["xs"])
        dz = probe(candidate["zc"], 14, upper, candidate["zs"])
        candidate["full"] = min(dx, dz)

    leaders = sorted(finalists, key=score, reverse=True)[:2]
    for candidate in leaders:
        candidate["da"] = probe(
            candidate["ac"], 48, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 64, candidate["db"])
        upper = min(candidate["da"], candidate["db"],
                    candidate["full"])
        dx = probe(candidate["xc"], 24, upper, candidate["xs"])
        dz = probe(candidate["zc"], 24, upper, candidate["zs"])
        candidate["full"] = min(dx, dz)

    winner = max(finalists, key=score)
    assert css_ok(winner["hx"], winner["hz"])
    assert winner["k"] == 168
    assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
    assert all(len(cell) == 3 and len(set(cell)) == 3
               for M in (winner["A"], winner["B"])
               for row in M for cell in row)
    return {"idx": idx, "card": card,
            "cells": {"A": winner["A"], "B": winner["B"]},
            "screen": score(winner), "six": winner["six"],
            "norm_clean": winner["norm_clean"],
            "hx": winner["hx"], "hz": winner["hz"]}


def _norm_tanner_signature(rows):
    """Permutation invariant of the weighted check-intersection graph.

    Unequal signatures exclude isomorphism. Equality is conservatively
    rejected; it is not asserted to prove isomorphism.
    """
    size = len(rows)
    adjacency = [0] * size
    overlaps = [[0] * size for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i][j] = overlaps[j][i] = overlap
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    signatures = []
    for i, row in enumerate(rows):
        histogram = {}
        for j in range(size):
            if i == j:
                continue
            key = (overlaps[i][j],
                   (adjacency[i] & adjacency[j]).bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        signatures.append(
            (row.bit_count(), tuple(sorted(histogram.items()))))
    return tuple(sorted(signatures))


from functools import lru_cache


@lru_cache(maxsize=1)
def _norm_design_bank(G):
    """Independent card optimizations, coupled only by duplicate rejection."""
    bank, fingerprints = [], set()
    for idx in range(N_SLOTS):
        chosen = None
        for retry in range(3):
            rng = random.Random(
                202610310030 + 104729 * idx + 100003 * retry)
            try:
                item = _norm_optimize(rng, G, idx)
            except ValueError:
                continue
            fingerprint = tuple(sorted((
                _norm_tanner_signature(item["hx"]),
                _norm_tanner_signature(item["hz"]),
            )))
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            # Keep only emitted cells and selection metadata in the cache.
            del item["hx"], item["hz"]
            chosen = item
            break
        bank.append(chosen)

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct norm designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return a fresh optimized card, never a transported orbit copy."""
    item = _norm_design_bank(G)[idx]
    if item is None:
        raise ValueError("no invariant-distinct full-rank design for card")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected cubic cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Order-42 binomials preserve the cubic bed's distance advantage; every distinct binomial-profile parent slot read d2s=5, while all five measured codes used five trinomials.
# REFUTED: Component-kernel selection alone removes the parent's mixed-sector bottleneck; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Cubic pair packing, six-cycle ranking and joint dependency probes exclude weight-nine mixed-sector failures; current-parent slot 21 still had d30s=9 with an X logical touching Arow0/Brow0, while four of five measured leaders remained below 12.
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
