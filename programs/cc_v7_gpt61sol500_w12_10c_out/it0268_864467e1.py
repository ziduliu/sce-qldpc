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
CONCEPT = ("Dic8, order 32, A=(1,3), B=(2,5): a controlled four-dimensional B cokernel with cubic constituent columns, exact short-dependency screening and verified logical searches should exceed the 0.350 frontier fitness 23.5; this is a falsifiable measured-score hypothesis.")
GROUP = ("L2 order repair within the unchanged dicyclic family. The new size factor 17 permits q<=35, so order 32 is the largest admissible dicyclic lift. Dic8 is a 2-group: odd-augmentation elements of F2[Dic8] are units. This supports explicit component-rank control, but every emitted rank and CSS condition is checked with verified helpers. The parent's Dic21 distance readings do not transfer automatically.")
SHAPES = ("Target 0.350 with (1,3)x(2,5): n=544, kb=192, guaranteed rate 6/17, and component variable/check counts 96/32 and 160/64. The prescribed B cokernel targets k=200, rate 0.367647, within [0.325,0.375). Parent weight-ten X witnesses occupied Acol/Bcol blocks, while slot 21's weight-nine witness also touched Arow0/Brow0. Screen constituents and both complete logical quotients.")
FILLING = ("Twenty cards cross five A trinomial pair-order recipes with four B architectures. A has three mixed-coset trinomials. B has four top-row order-eight binomials, four bottom-row monomials, and one bottom-only trinomial: every constituent column remains cubic, HX weight is 12 and HZ maximum weight is 11. Top binomial images equal the augmentation ideal of normal C8=<x^2>, targeting rank(B)=60 and an eight-qubit rank surplus. Optimize fresh labels, exactly screen anchored constituent dependencies through eight, deepen actual kernel searches, and verify full logical witnesses and stabilizer dressings. Unequal uncoloured Tanner fingerprints separate emitted designs. Search readings remain upper bounds, not distance certificates.")
FAMILY_NAME = "r035_Dic8_13x25_C8ideal_rank200_short8_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Same dicyclic rule: Dic8, x^16=1, y^2=x^8, y*x*y^-1=x^-1.
    if q != 32:
        raise ValueError("Dic8 has order 32")
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        a, b = g
        return ((a + 8 * (b // 2)) % 16, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 8*b*d) % 16,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 16, 0) if b == 0 else ((a + 8) % 16, 1)

    def idx(g):
        a, b = red(g)
        return 16*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
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


def _retired_dic21_context(rng, G, idx):
    """Inactive historical optimizer; the new shape never calls this.

    Twenty cubic pair-order cards, followed by joint witness screening.

    The older count plans and generator are superseded here. Matrix
    syndromes, ranks and stabilizer spaces use only verified lifts.
    """
    from math import gcd
    card = idx % 20
    a_cards = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3))
    b_cards = ((1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3))
    ad = a_cards[card // 4]
    bd = b_cards[card % 4]
    # gcd(d,42)=1,2,3 gives rotation-pair orders 42,21,14.
    steps = {
        divisor: [d for d in range(1, 42) if gcd(d, 42) == divisor]
        for divisor in (1, 2, 3)
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

    def score(candidate):
        d = min(candidate["da"], candidate["db"], candidate["full"])
        return (candidate["k"] * d * d / candidate["n"], d,
                candidate["da"] + candidate["db"],
                -candidate["six"][1], -sum(candidate["six"]))

    # First optimize genuine graph structure. All retained matrices have
    # cubic columns and no component four-cycles, not merely high search
    # readings from a short information-set budget.
    raw = []
    for attempt in range(64):
        A, B = packed(ad), packed(bd)
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br, ac, bc = _component_data(hx, hz, G.q)
        if _component_cycles(ar) or _component_cycles(br):
            continue
        ta, tb = six_cycles(ac), six_cycles(bc)
        structure = (2 * tb + ta, max(ta, tb), tb, ta)
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
            "ac": ac, "bc": bc, "six": (ta, tb),
            "structure": structure, "full": n + 1,
        }
        raw.append(candidate)
        raw.sort(key=lambda c: c["structure"])
        del raw[12:]

    pool = []
    for candidate in raw:
        hx, hz, n = candidate["hx"], candidate["hz"], candidate["n"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 275 * n <= 1000 * k < 325 * n:
            continue
        candidate["k"] = k
        if min(_short_floor(candidate["ac"]),
               _short_floor(candidate["bc"])) < 5:
            continue
        candidate["da"] = probe(candidate["ac"], 10)
        candidate["db"] = probe(candidate["bc"], 14)
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
    assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
    assert all(len(cell) == 3 and len(set(cell)) == 3
               for M in (winner["A"], winner["B"])
               for row in M for cell in row)
    return {"idx": idx, "card": card,
            "cells": {"A": winner["A"], "B": winner["B"]},
            "screen": score(winner), "six": winner["six"]}


from functools import lru_cache
from collections import Counter


_A32_RECIPES = (
    (1, 1, 1), (1, 1, 2), (1, 2, 2),
    (1, 2, 4), (1, 1, 4),
)
_SA32 = [[3, 3, 3]]
_SB32 = [[2, 2, 2, 2, 0], [1, 1, 1, 1, 3]]


def _s32_basis(rows):
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


def _s32_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


@lru_cache(maxsize=1)
def _s32_atlas(G):
    """Compile unit blocks only from the verified construction."""
    q = G.q
    mask = (1 << q) - 1
    za = [[[] for _ in range(3)]]
    zb = [[[] for _ in range(5)] for _ in range(2)]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)
    return atlas


def _s32_component(G, atlas, side, grid):
    """Linear superposition of scaffold-generated unit blocks."""
    q = G.q
    width = len(grid[0])
    cols = [0] * (width * q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[side][G.idx(g)]
                for t, syndrome in enumerate(block):
                    cols[j*q+t] ^= syndrome << (i*q)
    return cols


def _s32_cycles(cols):
    pairs, total = {}, 0
    for syndrome in cols:
        positions = []
        while syndrome:
            bit = syndrome & -syndrome
            positions.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                previous = pairs.get(key, 0)
                total += previous
                pairs[key] = previous + 1
    return total


def _s32_triple(rng, divisor):
    from math import gcd
    a, b = rng.randrange(16), rng.randrange(2)
    delta = rng.choice(
        [d for d in range(1, 16) if gcd(d, 16) == divisor])
    return [(a, b), ((a+delta) % 16, b),
            (rng.randrange(16), 1-b)]


def _s32_draw(rng, side, recipe, mode):
    if side == "A":
        return [[_s32_triple(rng, d) for d in recipe]]

    # Multiplicity patterns 2+2 and 3+1 among the two primitive
    # inverse-pair classes of C8 are different structural policies.
    gaps = ([2, 2, 6, 6] if mode < 2 else [2, 2, 2, 6])
    rng.shuffle(gaps)
    top, bottom = [], []
    for gap in gaps:
        a, b = rng.randrange(16), rng.randrange(2)
        delta = rng.choice((gap, 16-gap))
        top.append([(a, b), ((a+delta) % 16, b)])
        bottom.append([(rng.randrange(16), rng.randrange(2))])
    top.append([])
    bottom.append(_s32_triple(rng, 1 if mode % 2 == 0 else 2))
    return [top, bottom]


def _s32_decoder(cols, height):
    """Exact bounded decoding after component exclusion through four."""
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
            incidence[bit.bit_length()-1].append(i)
            value ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            assert value not in tails
            tails[value] = bits[i] | bits[j]

    def decode(syndrome, limit, forbidden):
        answer = None

        def visit(value, remaining, word):
            nonlocal answer
            if answer is not None:
                return
            if not value:
                answer = word
                return
            if value.bit_count() > 3*remaining:
                return
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    answer = word | tail
                return
            check = (value & -value).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & value).bit_count())
            for j in choices:
                if not bits[j] & (word | forbidden):
                    visit(value ^ cols[j], remaining-1, word | bits[j])
                    if answer is not None:
                        return

        visit(syndrome, limit, 0)
        return answer

    return decode


def _s32_short8(cols, q, height):
    """Exhaust constituent supports through eight, using block anchors.

    The constituent regular action is transitive within every block.
    This anchor reduction is not assumed for the complete quantum code.
    Nine is an exclusion sentinel, not a sampled distance reading.
    """
    decode = _s32_decoder(cols, height)
    for bound in (6, 8):
        for anchor in range(0, len(cols), q):
            completion = decode(cols[anchor], bound-1, 1 << anchor)
            if completion is not None:
                word = completion | (1 << anchor)
                syndrome, remaining = 0, word
                while remaining:
                    bit = remaining & -remaining
                    syndrome ^= cols[bit.bit_length()-1]
                    remaining ^= bit
                assert syndrome == 0
                return word.bit_count(), word
    return 9, 0


def _s32_probe(cols, rng, trials, seeds=(), stabilizers=None,
               initial=None):
    """Retain true kernel words; full-code readings exclude stabilizers."""
    span = None if stabilizers is None else _s32_basis(stabilizers)
    generators = [] if stabilizers is None else list(stabilizers)
    best = len(cols)+1 if initial is None else initial
    memory = sorted({w for w in seeds if w}, key=int.bit_count)[:40]
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if span is None or _s32_outside(word, span):
            best = word.bit_count()

    for word in memory:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        assert syndrome == 0
        observe(word)

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

        reducers = sorted(set(dependencies+memory),
                          key=int.bit_count)[:36]
        for sweep in range(2):
            changed = False
            for i, word in enumerate(dependencies):
                weight = word.bit_count()
                for direction in reducers:
                    candidate = word ^ direction
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                        changed = True
                dependencies[i] = word
                observe(word)
            reducers = sorted(set(dependencies+memory),
                              key=int.bit_count)[:36]
            if not changed:
                break

        discoveries = set(reducers+memory)
        short = sorted(discoveries, key=int.bit_count)[:40]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    if candidate.bit_count() <= best+4:
                        discoveries.add(candidate)

        if generators:
            logical = [w for w in discoveries
                       if _s32_outside(w, span)]
            for word in sorted(logical, key=int.bit_count)[:3]:
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in generators),
                        key=lambda x: x[0])[:12]
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
                    observe(word)
                discoveries.add(word)

        memory = sorted(discoveries, key=int.bit_count)[:40]
        if best <= 4:
            break
    return best, tuple(memory)


def _s32_small_logical(cols, stabilizers):
    """Exact logical-existence test through weight four."""
    span = _s32_basis(stabilizers)
    bits = [1 << i for i in range(len(cols))]
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif _s32_outside(bits[i] ^ previous, span):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif _s32_outside(word ^ previous, span):
                return True
    return False


def _s32_pool(rng, G, atlas, side, recipe, mode, quota):
    pool, seen = [], set()

    def merit(item):
        return (min(item["d"], 14), int(item.get("clean8", False)),
                -item["cycles"])

    def admit(grid):
        labels = tuple(tuple(sorted(cell))
                       for row in grid for cell in row)
        if labels in seen:
            return
        seen.add(labels)
        cols = _s32_component(G, atlas, side, grid)
        assert all(s.bit_count() == 3 for s in cols)
        cycles = _s32_cycles(cols)
        if side == "A" and cycles:
            return
        if _short_floor(cols) < 5:
            return
        expected = G.q if side == "A" else 60
        if gf2_rank(cols) != expected:
            return
        d, words = _s32_probe(cols, rng, 2)
        item = dict(grid=grid, cols=cols, cycles=cycles,
                    d=d, words=words)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[10:]

    for attempt in range(quota):
        admit(_s32_draw(rng, side, recipe, mode))

    # Actual voltage repairs stay inside the card's architecture.
    for step in range(quota//3):
        if not pool:
            admit(_s32_draw(rng, side, recipe, mode))
            continue
        parent = rng.choice(pool[:min(4, len(pool))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        replacement = _s32_draw(rng, side, recipe, mode)
        if side == "A":
            j = rng.randrange(3)
            grid[0][j] = replacement[0][j]
        elif step % 5 == 0:
            grid[1][4] = replacement[1][4]
        else:
            j = rng.randrange(4)
            # Replacing the bottom monomial changes coupling without
            # altering the prescribed top C8 ideal or gap multiplicities.
            grid[1][j] = replacement[1][j]
        admit(grid)

    for item in pool:
        floor, witness = _s32_short8(
            item["cols"], G.q, G.q if side == "A" else 2*G.q)
        item["clean8"] = floor == 9
        seeds = item["words"] + ((witness,) if witness else ())
        start = min(item["d"], floor) if witness else item["d"]
        item["d"], item["words"] = _s32_probe(
            item["cols"], rng, 24, seeds, initial=start)
        if witness:
            item["d"] = min(item["d"], witness.bit_count())

    pool.sort(key=merit, reverse=True)
    return pool[:4]


def _s32_fingerprint(hx, hz, n):
    """Invariant of the combined uncoloured check/qubit incidence graph.

    Unequal signatures prove nonisomorphism; equal signatures are
    conservatively rejected rather than claimed to prove isomorphism.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    size = len(rows)
    adjacency = [0] * size
    overlap = [[0] * size for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = value
            if value:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = []
    for i in range(size):
        histogram = Counter(
            (weights[j], overlap[i][j],
             (adjacency[i] & adjacency[j]).bit_count())
            for j in range(size) if j != i)
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    degrees = [0] * n
    for source in rows:
        while source:
            bit = source & -source
            degrees[bit.bit_length()-1] += 1
            source ^= bit
    return (n, tuple(sorted(Counter(degrees).items())),
            tuple(sorted(Counter(profiles).items())))


def _s32_embeddings(words, side, q):
    mask = (1 << q)-1
    if side == "B":
        return [word << (j*5*q) for word in words[:16]
                for j in range(3)]
    return [
        sum(((word >> (j*q)) & mask) << ((j*5+beta)*q)
            for j in range(3))
        for word in words[:16] for beta in range(5)
    ]


@lru_cache(maxsize=1)
def _s32_bank(G):
    """Independently optimize twenty rank-controlled structural cards."""
    assert G.q == 32
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    assert _grid_weights(_SA32, _SB32) == 12
    assert _no_dead_lines(_SA32, _SB32)
    atlas = _s32_atlas(G)
    bank, emitted = [], set()
    q, mask = G.q, (1 << G.q)-1

    def merit(item):
        d = min(item["da"], item["db"], item["dx"], item["dz"], 14)
        return (item["k"]*d*d/item["n"], d,
                item["da"]+item["db"], -item["cycles"])

    for idx in range(N_SLOTS):
        recipe = _A32_RECIPES[(idx % 20)//4]
        mode = idx % 4
        rng = random.Random(BASE_SEED*100003 + 35032200 + 104729*idx)
        winner = None

        for retry in range(2):
            ap = _s32_pool(rng, G, atlas, "A", recipe, mode,
                          112+48*retry)
            bp = _s32_pool(rng, G, atlas, "B", recipe, mode,
                          80+32*retry)
            proposals = []
            for aa in ap:
                for bb in bp:
                    d = min(aa["d"], bb["d"])
                    proposals.append(
                        ((d, aa["d"]+bb["d"],
                          -aa["cycles"]-bb["cycles"]), aa, bb))
            proposals.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for _, aa, bb in proposals:
                A, B = aa["grid"], bb["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or k != 200:
                    continue

                # Validate compiled component actions against final lifts.
                ar = [
                    sum(((row >> (j*5*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert _columns(ar, 3*q) == aa["cols"]
                assert _columns(br, 5*q) == bb["cols"]

                fp = _s32_fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                xc, zc = _columns(hz, n), _columns(hx, n)
                if _s32_small_logical(xc, hx):
                    continue
                if _s32_small_logical(zc, hz):
                    continue

                dx, xwords = _s32_probe(
                    xc, rng, 8,
                    _s32_embeddings(bb["words"], "B", q),
                    stabilizers=hx)
                dz, zwords = _s32_probe(
                    zc, rng, 8,
                    _s32_embeddings(aa["words"], "A", q),
                    stabilizers=hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    da=aa["d"], db=bb["d"], fp=fp,
                    cycles=aa["cycles"]+bb["cycles"]))
                if len(finalists) == 4:
                    break

            # Equal independent full-code budgets for all finalists.
            for item in finalists:
                dx, _ = _s32_probe(
                    item["xc"], rng, 20, item["xwords"],
                    stabilizers=item["hx"], initial=item["dx"])
                dz, _ = _s32_probe(
                    item["zc"], rng, 20, item["zwords"],
                    stabilizers=item["hz"], initial=item["dz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["k"] == 200
        assert 325*winner["n"] <= 1000*winner["k"] < 375*winner["n"]
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) <= W_MAX
        A, B = winner["A"], winner["B"]
        assert [[len(c) for c in row] for row in A] == _SA32
        assert [[len(c) for c in row] for row in B] == _SB32
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A, B) for row in grid for c in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={
                "A": tuple(tuple(tuple(c) for c in row) for row in A),
                "B": tuple(tuple(tuple(c) for c in row) for row in B)},
            k=winner["k"], card=(recipe, mode),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the fresh rank-controlled filling for this structural card."""
    selected = _s32_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rank-200 Dic8 filling passed this card")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return every cell of the new shapes without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Order-42 binomials preserve the cubic bed's distance advantage; every distinct binomial-profile parent slot read d2s=5, while all five measured codes used five trinomials.
# REFUTED: Component-kernel selection alone removes the parent's mixed-sector bottleneck; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Cubic pair packing and joint information-set probes suffice for five Dic21 readings >=12; the current measured leaders instead read 10,10,12,9,10, including sector-one weight-ten and mixed-sector weight-nine X witnesses.
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
