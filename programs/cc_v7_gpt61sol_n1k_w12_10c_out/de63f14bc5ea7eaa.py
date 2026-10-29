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
CONCEPT = ("Frozen A4 x C5, order 60, A=(1,3), B=(2,7): commuting "
           "order-ten V4 x C5 rings and quotient-balanced cubic columns "
           "should add verified check-rank deficits while retaining five "
           "provisional distance-eight readings, improving cell 0.450 "
           "fitness beyond 28.5220.")
GROUP = ("Keep the exact group definition and order 60 byte-for-byte "
         "unchanged. This bed repeatedly measured [1380,615,8]; longer "
         "orders did not produce ten. The normal V4 x C5 subgroup has "
         "index three, giving a concrete binary quotient-dependency "
         "mechanism. Verify its dependencies and full ranks on scaffold "
         "matrices rather than inferring distance from element orders.")
SHAPES = ("Keep A=(1,3), B=(2,7) byte-for-byte unchanged. Factor 23 gives "
          "n=1380, k>=600 and guaranteed rate 10/23=0.434783 in cell "
          "0.450; q_max=65 and the frozen group fixes q=60. Unequal "
          "components have 180 variables/60 checks and 420 variables/"
          "120 checks. Parent minima were entirely sector one: X words "
          "emphasized Acol1/2 across Bcol0..2, while Z families frequently "
          "used Bcol3/4 across all Acols. Change dependencies and labels, "
          "not these component lengths.")
FILLING = ("Retain cubic A cells, B degrees (2,2,2,3,3,3,3), B row "
           "totals nine and both CSS maxima twelve. Build two commuting "
           "order-ten rings generating the proper V4 x C5 subgroup; "
           "verify their shortest cycle on authoritative blocks. Every "
           "B cubic tail COLUMN has one term in each C3 quotient phase "
           "across its two cells, preserving two explicitly verified "
           "check dependencies. Twenty cards cross five A central-ratio "
           "coupling profiles with four B supports. A cubics also balance "
           "quotient phases; mutations preserve their coupling recipes. "
           "Exactly challenge A component words through six, retain "
           "genuine cancellation witnesses and independently deepen "
           "finalists. Select using authoritative full k and original-span "
           "logical challenges, with readings capped at eight. Native "
           "assembly, distinctness, weights, CSS and [0.425,0.475) remain "
           "gates. Neither component exclusions nor finite searches "
           "certify quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r045_A4xC5_13x27_quotientbalanced_subgroup_rings_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 60

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C5 requires order 60")
    # First four entries encode an even permutation; the fifth is central.
    even = [p for p in permutations(range(4))
            if sum(p[i] > p[j] for i in range(4)
                   for j in range(i+1, 4)) % 2 == 0]
    els = [p + (c,) for c in range(5) for p in even]
    assert len(even) == 12 and len(els) == q
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 5,)

    def mul(g, h):
        g, h = red(g), red(h)
        return tuple(g[h[i]] for i in range(4)) + ((g[4]+h[4]) % 5,)

    def inv(g):
        g = red(g)
        answer = [0]*4
        for i in range(4):
            answer[g[i]] = i
        return tuple(answer) + ((-g[4]) % 5,)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0,1,2,3,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Five ternary relative-order spectra, not automorphism images:
# mode 0: (5,10,10); mode 1: (5,5,5);
# mode 2: (10,10,11); mode 3: (5,5,11);
# mode 4: (2,5,10).
# Cross these with the count of order-five B binomial ratios.
_ODD_V = (1, 3, 7, 9)
_EVEN_V = (2, 4, 6, 8)


def _affine_trinomial(rng, G, mode):
    """Anchor one term; preserve the prescribed relative-order spectrum."""
    if mode == 0:
        v, w = rng.sample(_ODD_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode == 1:
        v, w = rng.sample(_EVEN_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode in (2, 3):
        pool = _ODD_V if mode == 2 else _EVEN_V
        g = (rng.randrange(1, 11), 0)
        h = (rng.randrange(11), rng.choice(pool))
    else:
        v = rng.choice(_ODD_V)
        g = (rng.randrange(11), v)
        h = (rng.randrange(11), (v + 5) % 10)
    cell = [G.e, g, h]
    assert len({G._key(x) for x in cell}) == 3
    return cell


def _affine_side(rng, G, side, card):
    """Twenty deliberate order-spectrum cards with a full-action bridge."""
    mode, five = card % 5, card // 5
    if side == "A":
        # The first column's odd multipliers bridge the index-two subgroup.
        return [_affine_trinomial(rng, G, 0),
                _affine_trinomial(rng, G, mode),
                _affine_trinomial(rng, G, mode)]
    cells = []
    for j in range(3):
        pool = _EVEN_V if j < five else _ODD_V
        cells.append([G.e, (rng.randrange(11), rng.choice(pool))])
    cells.append(_affine_trinomial(rng, G, 0))
    return cells


def _verified_single_component(G, cells, side):
    """Use the scaffold lift, with an empty dummy side for extraction only."""
    if side == "A":
        hx, _, _ = lifted_matrices(G, [cells], [[[]]])
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, [[[]]], [cells])
        rows = hz
    width = len(cells) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _component_kernel4(rows, width):
    """Exact exclusion of component kernel words of weights one through four."""
    columns = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            columns[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            syndrome = a ^ b
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _component_geometry(rows):
    """Connectivity and exact four/six cycles of a verified bipartite lift."""
    count = len(rows)
    common = [[0] * count for _ in range(count)]
    upper = [[] for _ in rows]
    adjacent = [0] * count
    four = 0
    for a in range(count):
        for b in range(a + 1, count):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                adjacent[a] |= 1 << b
                adjacent[b] |= 1 << a
                four += overlap * (overlap - 1) // 2

    reached = frontier = 1
    while frontier:
        neighbors = 0
        while frontier:
            bit = frontier & -frontier
            neighbors |= adjacent[bit.bit_length() - 1]
            frontier ^= bit
        frontier = neighbors & ~reached
        reached |= frontier
    if reached != (1 << count) - 1:
        return None

    six = 0
    for a, neighbors in enumerate(upper):
        for t, b in enumerate(neighbors):
            ab = common[a][b]
            for c in neighbors[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    six += ab * ac * bc - triple * (ab + ac + bc - 2)
    return four, six


def _kernel_witness_reading(rows, width, rng, trials):
    """Randomized information-set search for genuine classical kernel words.

    A high reading means this search missed lighter words. These are
    component witnesses, not certified quantum logical distances.
    """
    best = width + 1
    for _ in range(trials):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapping = [1 << p for p in permutation]
        pivots = {}
        for source in rows:
            row = 0
            while source:
                bit = source & -source
                row ^= mapping[bit.bit_length() - 1]
                source ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ordered = sorted(pivots)
        for at, p in enumerate(ordered):
            bit = 1 << p
            reduced = pivots[p]
            for higher in ordered[at + 1:]:
                if pivots[higher] & bit:
                    pivots[higher] ^= reduced

        words = []
        for f in range(width):
            if f in pivots:
                continue
            bit = 1 << f
            word = bit
            for p in ordered:
                if pivots[p] & bit:
                    word |= 1 << p
            best = min(best, word.bit_count())
            words.append(word)
        words.sort(key=int.bit_count)
        shortlist = words[:24]
        for i, word in enumerate(shortlist):
            for other in shortlist[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 5:
            break
    return best


def _screened_component_pool(rng, G, side, card):
    """Compare independent fillings of one prescribed order-spectrum card."""
    pool = []
    width = (3 if side == "A" else 4) * G.q
    for trial in range(96):
        cells = _affine_side(rng, G, side, card)
        rows = _verified_single_component(G, cells, side)
        geometry = _component_geometry(rows)
        if geometry is None:
            continue
        if not _component_kernel4(rows, width):
            continue
        # Full component rank avoids selecting rank-deficient fallbacks.
        if gf2_rank(rows) != G.q:
            continue
        pool.append((geometry, trial, cells, rows))
        if len(pool) >= 24:
            break
    if not pool:
        raise ValueError("no connected full-rank component passed kernel-4")

    pool.sort(key=lambda record: (record[0], record[1]))
    ranked = []
    for geometry, trial, cells, rows in pool[:6]:
        reading = _kernel_witness_reading(rows, width, rng, 8)
        key = (reading, -geometry[0], -geometry[1])
        ranked.append((key, trial, cells, rows, geometry))
    ranked.sort(key=lambda record: record[0], reverse=True)

    finalists = []
    for key, trial, cells, rows, geometry in ranked[:3]:
        reading = min(key[0], _kernel_witness_reading(
            rows, width, rng, 32))
        final_key = (reading, -geometry[0], -geometry[1])
        finalists.append((final_key, trial, cells))
    finalists.sort(key=lambda record: record[0], reverse=True)
    return finalists


def _legacy_even_ctx(rng, G, idx):
    """Unused historical affine experiment; NEW_CTX below replaces it."""
    from itertools import combinations

    card = idx % 20
    cohort = idx // 20
    q = G.q
    sa, sb = [[2, 2, 2]], [[2, 2, 2, 4]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    # Four quotient-voltage profiles crossed with five quartic geometries.
    # These change relations among voltages, not merely their labels.
    profiles = ((1, 1, 1), (1, 1, 3), (1, 3, 3), (3, 3, 3))
    quartics = ((0, 0, 1), (0, 2, 3), (2, 4, 1),
                (2, 6, 3), (4, 8, 1))
    pa = card // 5
    pb = (3 - pa + cohort) % 4
    qm = (card % 5 + cohort) % 5

    def columns(rows, width):
        out = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                out[bit.bit_length() - 1] |= 1 << r
                source ^= bit
        return out

    def planned(side):
        profile = profiles[pa if side == "A" else pb]
        cells = []
        for exponent in profile:
            # Reversing an individual edge does not create a new design;
            # diversity comes from the profiles and independently optimized
            # relations among all cells.
            b = exponent if rng.randrange(2) else (-exponent) % 10
            cells.append([G.e, (rng.randrange(11), b)])
        if side == "B":
            e1, e2, odd = quartics[qm]
            for _ in range(100):
                cell = [G.e, (rng.randrange(11), e1),
                        (rng.randrange(11), e2),
                        (rng.randrange(11), odd)]
                if len({G._key(g) for g in cell}) == 4:
                    cells.append(cell)
                    break
            else:
                raise ValueError("quartic distinctness failed")
        return cells

    def detected_six(cols):
        """Verified six-word collisions among connected triples.

        This is a selection statistic, not exhaustive six-word counting.
        Exact smaller-word exclusion is performed separately.
        """
        incidence = {}
        for v, syndrome in enumerate(cols):
            while syndrome:
                bit = syndrome & -syndrome
                incidence.setdefault(bit.bit_length() - 1, []).append(v)
                syndrome ^= bit
        adjacency = [set() for _ in cols]
        for variables in incidence.values():
            for v in variables:
                adjacency[v].update(u for u in variables if u != v)
        triples = set()
        for middle, neighbors in enumerate(adjacency):
            for a, b in combinations(sorted(neighbors), 2):
                triples.add(tuple(sorted((a, middle, b))))
        seen, found = {}, set()
        for a, b, c in sorted(triples):
            syndrome = cols[a] ^ cols[b] ^ cols[c]
            word = (1 << a) | (1 << b) | (1 << c)
            previous = seen.get(syndrome)
            if previous is not None:
                witness = previous ^ word
                if witness.bit_count() == 6:
                    found.add(witness)
            else:
                seen[syndrome] = word
        return len(found)

    def component_pool(side):
        width = (3 if side == "A" else 4) * q
        pool, signatures = [], set()
        charge = sum(1 << r for r, g in enumerate(G.elements)
                     if g[1] & 1)
        for trial in range(192):
            cells = planned(side)
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in cells)
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = _verified_single_component(G, cells, side)
            if gf2_rank(rows) != q - 1:
                continue
            if not _component_kernel4(rows, width):
                continue
            cols = columns(rows, width)
            # Every component variable has odd charge, even though its
            # ordinary column weight is even. Thus odd kernel weights
            # are impossible, including weight five.
            if not all((c & charge).bit_count() & 1 for c in cols):
                continue
            geometry = _component_geometry(rows)
            if geometry is None:
                continue
            overlap = max((rows[a] & rows[b]).bit_count()
                          for a in range(q) for b in range(a))
            if side == "B" and overlap > 4:
                continue
            six = detected_six(cols)
            # Cohorts deliberately trade witness multiplicity against
            # Tanner geometry; no cohort is produced by relabeling.
            if cohort % 3 == 0:
                key = (six, geometry[0], geometry[1], overlap)
            elif cohort % 3 == 1:
                key = (six, overlap, geometry[0], geometry[1])
            else:
                key = (six + geometry[0], six, overlap, geometry[1])
            pool.append((key, trial, cells, rows, overlap))
            if len(pool) >= 12:
                break
        if not pool:
            raise ValueError("no parity-controlled component passed screens")
        pool.sort(key=lambda item: (item[0], item[1]))
        return pool[:4]

    def full_five_free(hx, hz, n, brows):
        """Exact exclusion through five using verified syndromes.

        Full weight <=4 is exhaustively excluded. HZ has an odd charge
        on EVERY column, prohibiting all odd kernel weights.

        For HX, sector-1 columns have charge one and weight two;
        sector-2 columns have charge zero and weight ten. A weight-five
        kernel must therefore contain 1, 3 or 5 sector-2 variables.
        One cannot cancel with four weight-two columns.
        Three have XOR weight >=30-2*(4+4+4)=6, exceeding the four
        bits supplied by two sector-1 columns.
        Five sector-2 variables cannot be a kernel: the verified B
        component has rank q-1 and its transpose kernel is exactly
        the all-ones vector of weight q.
        """
        if not _component_kernel4(hx, n):
            return False
        if not _component_kernel4(hz, n):
            return False
        n1 = 12 * q
        xcharge = sum(1 << (beta * q + t)
                      for beta in range(4)
                      for t, g in enumerate(G.elements) if g[1] & 1)
        zcharge = sum(1 << (j * q + t)
                      for j in range(3)
                      for t, g in enumerate(G.elements) if g[1] & 1)
        xc, zc = columns(hx, n), columns(hz, n)
        if not all((c & zcharge).bit_count() & 1 for c in zc):
            return False
        if not all(c.bit_count() == 2 and
                   (c & xcharge).bit_count() & 1 for c in xc[:n1]):
            return False
        if not all(c.bit_count() == 10 and
                   not ((c & xcharge).bit_count() & 1)
                   for c in xc[n1:]):
            return False
        # Check the claimed transpose correspondence on actual matrices,
        # rather than relying on an independently implemented lift.
        if xc[n1:] != brows:
            return False
        if gf2_rank(brows) != q - 1:
            return False
        if any(sum((row >> c) & 1 for row in brows) & 1
               for c in range(4 * q)):
            return False
        return max((a & b).bit_count()
                   for i, a in enumerate(brows) for b in brows[:i]) <= 4

    for restart in range(3):
        aa, bb = component_pool("A"), component_pool("B")
        pairs = []
        for ak, at, acells, arows, ao in aa:
            for bk, bt, bcells, brows, bo in bb:
                key = (ak[0] + bk[0], max(ak[0], bk[0]),
                       ak[1:] + bk[1:])
                pairs.append((key, at, bt, acells, bcells, brows))
        pairs.sort(key=lambda item: item[:3])
        survivors = []
        for quality, at, bt, acells, bcells, brows in pairs:
            A, B = [acells], [bcells]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 425*n <= 1000*k < 475*n):
                continue
            if not full_five_free(hx, hz, n, brows):
                continue
            assert n == 1430
            assert max(row.bit_count() for row in hx + hz) <= 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            survivors.append((k, quality, A, B))
            if len(survivors) >= 3:
                break
        if survivors:
            survivors.sort(key=lambda item: (-item[0], item[1]))
            k, quality, A, B = survivors[0]
            return {"idx": idx, "card": card,
                    "cells": {"A": A, "B": B},
                    "k": k, "quality": quality,
                    "screen": "verified kernel weights 1..5 excluded"}
    raise ValueError("no all-even full lift passed kernel-five/rate screens")


def NEW_CTX(rng, G, idx):
    """Twenty quotient-balanced coupling/support cards with exact rank pairing."""
    from itertools import combinations
    from math import exp

    q = G.q
    assert q == 60
    mask = (1 << q)-1
    card, cohort = idx % 20, idx // 20
    composition, model = card % 5, card // 5
    diagnostic_seed = rng.randrange(1 << 62) ^ (
        0xA4C520327 + 15485863*(idx+1))
    rng = random.Random(diagnostic_seed ^ 0xC3B2A6)
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        # Cache group operations, not a separately implemented lift.
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    raw_a, _, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    _, raw_b, _ = lifted_matrices(
        probe, [[[]]], [[[g] for g in range(q)]])
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(cols, word):
        value = 0
        for v in bits(word):
            value ^= cols[v]
        return value

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")

    def generates(a, b):
        reached = {ident}
        todo = [ident]
        for x in todo:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    todo.append(z)
        return len(reached) == q

    # Card compositions constrain relations, not automorphism images.
    spectra = ((10,15,15), (15,15,15), (3,10,15),
               (3,15,15), (2,15,15))
    libraries = [[] for _ in spectra]
    general = []
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident or not generates(a, b):
                continue
            triple = (ident, a, b)
            spectrum = tuple(sorted((
                orders[a], orders[b], orders[table[inverses[a]][b]])))
            general.append((spectrum, triple))
            if spectrum in spectra:
                libraries[spectra.index(spectrum)].append(triple)
    assert general
    # Enrichment is deliberate, but an empty preferred catalog is not
    # allowed to destroy every slot.
    for kind, target in enumerate(spectra):
        if not libraries[kind]:
            ranked = sorted(general, key=lambda item: sum(
                abs(x-y) for x, y in zip(item[0], target)))
            libraries[kind] = [item[1] for item in ranked[:24]]

    # Construct the actual normal subgroup and its C3 quotient using
    # the frozen group operations. No independently written lift is used.
    normal = [g for g in range(q) if orders[g] in (1,2,5,10)]
    assert len(normal) == 20
    assert all(table[a][b] in normal for a in normal for b in normal)
    quotient_generator = next(g for g in range(q) if orders[g] == 3)
    quotient_phase = [-1]*q
    power = ident
    for phase in range(3):
        for h in normal:
            g = table[h][power]
            assert quotient_phase[g] == -1
            quotient_phase[g] = phase
        power = table[power][quotient_generator]
    assert power == ident and all(p >= 0 for p in quotient_phase)
    assert all(
        quotient_phase[table[a][b]] ==
        (quotient_phase[a]+quotient_phase[b]) % 3
        for a in range(q) for b in range(q))
    phase_pools = [
        [g for g in range(q) if quotient_phase[g] == phase]
        for phase in range(3)]
    central_coordinate = [int(g[4]) % 5 for g in G.elements]

    def subgroup_size(a, b):
        reached, todo = {ident}, [ident]
        for x in todo:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    todo.append(z)
        return len(reached)

    # Each functional has even weight in the three quotient coordinates.
    # Repeat it in BOTH B check-block rows. Balanced cubic columns and
    # phase-zero rings annihilate both independent functionals.
    b_functionals = [
        sum(1 << (i*q+r)
            for i in range(2) for r in range(q)
            if quotient_phase[r] in selected)
        for selected in ((0,1), (0,2))]
    assert gf2_rank(b_functionals) == 2

    def record(grid, side):
        """GF2 superposition of monomial blocks from the verified helper."""
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    action = blocks[side][g]
                    for t, value in enumerate(action):
                        rows[i*q+t] ^= value << (j*q)
        cols = [0]*(nc*q)
        for r, row in enumerate(rows):
            for v in bits(row):
                cols[v] |= 1 << r

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for col in cols:
            checks = list(bits(col))
            degree = len(checks)
            forced += degree*(degree-1)*(degree-2)//6
            for r in checks:
                adjacency[r] |= col ^ (1 << r)
            for a, b in combinations(checks, 2):
                key = (a, b)
                four += uses.get(key, 0)
                uses[key] = uses.get(key, 0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        # Projection excess is only a cycle-selection heuristic.
        excess = max(0, triangles-forced)
        base = (500*(1+model)*four + (1+cohort % 3)*excess)
        return {"grid": copy_grid(grid), "rows": rows,
                "cols": cols, "base": base, "cycles": (four, excess)}

    def backbone_word(rec):
        """Exact shortest cycle word of the verified degree-two component."""
        cols = rec["cols"]
        adjacency = [[] for _ in rec["rows"]]
        for v, col in enumerate(cols):
            checks = list(bits(col))
            assert len(checks) == 2
            a, b = checks
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))
        best = None
        for root in range(len(adjacency)):
            distance = [-1]*len(adjacency)
            paths = [0]*len(adjacency)
            parent_edge = [-1]*len(adjacency)
            distance[root] = 0
            todo = [root]
            for a in todo:
                for b, edge in adjacency[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a]+1
                        paths[b] = paths[a] ^ (1 << edge)
                        parent_edge[b] = edge
                        todo.append(b)
                    elif edge != parent_edge[a] and edge != parent_edge[b]:
                        word = paths[a] ^ paths[b] ^ (1 << edge)
                        if word and (best is None or
                                     word.bit_count() < best.bit_count()):
                            best = word
        assert best is not None and syndrome(cols, best) == 0
        return best

    # Commuting rings have a commutator cycle of weight eight, but their
    # individual cycles have weight ten. Require generation of the full
    # order-20 subgroup, not a smaller disconnected proposal subgroup.
    ring_elements = [g for g in normal if orders[g] == 10]
    ring_pairs = [
        (a,b) for at, a in enumerate(ring_elements)
        for b in ring_elements[:at]
        if subgroup_size(a,b) == 20]
    assert ring_pairs
    backbone_options = []
    for a, b in ring_pairs:
        assert table[a][b] == table[b][a]
        if (composition+model+cohort) & 1:
            a, b = b, a
        grid = [[[ident,a], [], [ident]],
                [[], [ident,b], [ident]]]
        rec = record(grid, "B")
        word = backbone_word(rec)
        assert all(
            (functional & col).bit_count() % 2 == 0
            for functional in b_functionals for col in rec["cols"])
        if word.bit_count() >= 8:
            backbone_options.append((word.bit_count(), a, b, grid, word))
    assert backbone_options
    best_girth = max(item[0] for item in backbone_options)
    (backbone_distance, ring_a, ring_b,
     skeleton, skeleton_word) = rng.choice([
         item for item in backbone_options if item[0] == best_girth])
    assert backbone_distance == 8

    profiles = (
        ((2,1), (2,1), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2)),
        ((3,0), (3,0), (0,3), (0,3)),
        ((3,0), (1,2), (1,2), (1,2)),
    )
    # For an anchored A cubic [e,g,h], quotient phases are (0,1,2)
    # and central coordinates are (0,u,t*u). Independently varying u
    # changes cross-cell relations; the five coupled t-profiles are cards,
    # not group-automorphism images deliberately submitted as new codes.
    central_ratio_profiles = (
        (2,2,2), (2,2,3), (2,3,4), (0,2,3), (1,2,4))
    aplan = (0,1,2)
    bcounts = [
        tuple(len(skeleton[i][j]) for i in range(2))
        for j in range(3)
    ] + list(profiles[model])
    desired_orders = ((15,), (3,15), (15,), (3,15))[model]
    planned_b = [[bcounts[j][i] for j in range(7)] for i in range(2)]
    assert [sum(row) for row in planned_b] == [9, 9]
    assert _grid_weights([[3,3,3]], planned_b) == 12
    assert _no_dead_lines([[3,3,3]], planned_b)

    def draw_cell(count, kind, side, phases=None):
        """Draw new relative supports while preserving binding quotient cards."""
        if not count:
            return []
        if side == "A":
            assert count == 3
            ratio = central_ratio_profiles[composition][kind]
            for _ in range(512):
                u = rng.randrange(1,5)
                g = rng.choice([
                    x for x in phase_pools[1]
                    if central_coordinate[x] == u])
                h = rng.choice([
                    x for x in phase_pools[2]
                    if central_coordinate[x] == ratio*u % 5])
                if generates(g,h):
                    return [ident,g,h]
            raise ValueError("balanced A coupling recipe exhausted")

        assert 1 <= count <= 3
        if phases is None:
            phases = tuple(rng.sample(range(3), count))
        assert len(phases) == count
        for _ in range(512):
            cell = [rng.choice(phase_pools[p]) for p in phases]
            if len(set(cell)) != count:
                continue
            root = inverses[cell[0]]
            if count == 2:
                if orders[table[root][cell[1]]] not in desired_orders:
                    continue
            elif count == 3:
                if not generates(table[root][cell[1]],
                                 table[root][cell[2]]):
                    continue
            return cell
        raise ValueError("balanced B phase recipe exhausted")

    def redraw_cell(old, kind, side):
        if side == "A":
            return draw_cell(len(old), kind, side)
        return draw_cell(
            len(old), kind, side,
            tuple(quotient_phase[g] for g in old))

    def term_variant(old, kind, side):
        """Change one term, preserving quotient balance and A central coupling."""
        if not old:
            return None
        proposal = list(old)
        at = rng.choice((1,2)) if side == "A" else rng.randrange(len(old))
        previous = old[at]
        choices = phase_pools[quotient_phase[previous]]
        if side == "A":
            choices = [
                g for g in choices
                if central_coordinate[g] == central_coordinate[previous]]
        proposal[at] = rng.choice(choices)
        if proposal == old or len(set(proposal)) != len(proposal):
            return None
        root = inverses[proposal[0]]
        if len(proposal) == 2:
            return (proposal if orders[table[root][proposal[1]]]
                    in desired_orders else None)
        if len(proposal) == 3 and not generates(
                table[root][proposal[1]], table[root][proposal[2]]):
            return None
        return proposal

    def fresh(side):
        if side == "A":
            return [[draw_cell(3, kind, side) for kind in aplan]]
        grid = copy_grid(skeleton)
        for j, counts in enumerate(profiles[model], 3):
            # Balance the WHOLE column, not each split cell separately.
            atoms = draw_cell(3, j % 3, "B", (0,1,2))
            rng.shuffle(atoms)
            first = counts[0]
            grid[0].append(atoms[:first])
            grid[1].append(atoms[first:])
            assert sorted(quotient_phase[g]
                          for i in range(2) for g in grid[i][j]) == [0,1,2]
        return grid

    def small_word(cols):
        """Exhaustive kernel exclusion through four; return a real witness."""
        singles = {}
        for v, col in enumerate(cols):
            word = 1 << v
            if not col:
                return word
            if col in singles:
                return word ^ singles[col]
            singles[col] = word
        pairs = {}
        for a, col in enumerate(cols):
            for b in range(a):
                value = col ^ cols[b]
                word = (1 << a) | (1 << b)
                old = singles.get(value)
                if old is not None:
                    return word ^ old
                old = pairs.get(value)
                if old is not None:
                    return word ^ old
                pairs[value] = word
        return None

    def exact_a_six(rec):
        """Exact classical A challenge through six after kernel-four exclusion.

        All A columns have weight three, excluding odd kernel weights.
        Right regular translation supplies one anchor per Acol. Unique
        singleton/pair completions and unbudgeted parity branching cover
        every anchored word through six; no quantum exclusion is claimed.
        """
        cols = rec["cols"]
        width = len(cols)
        assert width == 3*q and all(c.bit_count() == 3 for c in cols)
        singles = {value: 1 << v for v, value in enumerate(cols)}
        assert len(singles) == width and 0 not in singles
        pairs = {}
        incidence = [0]*len(rec["rows"])
        for v, value in enumerate(cols):
            for r in bits(value):
                incidence[r] |= 1 << v
            for u in range(v):
                key = value ^ cols[u]
                assert key not in singles and key not in pairs
                pairs[key] = (1 << v) | (1 << u)

        def visit(value, word, allowed):
            if not value:
                return word
            left = 6-word.bit_count()
            if left <= 0 or value.bit_count() > 3*left:
                return None
            one = singles.get(value, 0)
            if one and one & allowed == one:
                return word | one
            pair = pairs.get(value, 0)
            if left >= 2 and pair and pair & allowed == pair:
                return word | pair
            if left <= 2:
                return None
            chosen, smallest = 0, width+1
            for r in bits(value):
                available = incidence[r] & allowed
                count = available.bit_count()
                if not count:
                    return None
                if count < smallest:
                    chosen, smallest = available, count
            choices = sorted(bits(chosen), key=lambda v: (
                -(cols[v] & value).bit_count(), v))
            live = allowed
            for v in choices:
                bit = 1 << v
                live &= ~bit
                found = visit(value ^ cols[v], word | bit, live)
                if found is not None:
                    return found
            return None

        all_variables = (1 << width)-1
        for root in range(0, width, q):
            found = visit(cols[root], 1 << root,
                          all_variables ^ (1 << root))
            if found is not None:
                assert found.bit_count() == 6
                assert syndrome(cols, found) == 0
                return found
        return None

    def kernel_probe(rec, side, trials, seed):
        """Genuine witnesses, exact A six-challenge and retained cancellation."""
        local = random.Random(seed)
        cols = rec["cols"]
        width = len(cols)
        order = list(range(width))
        best_word = small_word(cols)
        if side == "A" and best_word is None:
            if "exact_six_word" not in rec:
                rec["exact_six_word"] = exact_a_six(rec)
            best_word = rec["exact_six_word"]
            rec["exact_A_kernel6_free"] = best_word is None
        if side == "B":
            assert syndrome(cols, skeleton_word) == 0
            if best_word is None or (
                    skeleton_word.bit_count() < best_word.bit_count()):
                best_word = skeleton_word

        retained = set(rec.get("witnesses", ()))

        def observe(word):
            nonlocal best_word
            if not word:
                return
            if best_word is None or word.bit_count() < best_word.bit_count():
                best_word = word
            if word.bit_count() <= 12:
                retained.add(word)
                if len(retained) > 96:
                    keep = sorted(retained,
                                  key=lambda w: (w.bit_count(), w))[:64]
                    retained.clear()
                    retained.update(keep)

        previous = rec.get("word")
        if previous is not None:
            assert syndrome(cols, previous) == 0
            observe(previous)
        for word in list(retained):
            assert syndrome(cols, word) == 0
            observe(word)
        if best_word is not None:
            observe(best_word)

        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = cols[v], 1 << v
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(
                set(basis) | retained,
                key=lambda w: (w.bit_count(), w))[:20]
            moves = basis + sorted(retained)
            for word in light:
                for other in moves:
                    observe(word ^ other)
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
            if best_word is not None and (
                    best_word.bit_count() <= 4 or
                    side == "A" and best_word.bit_count() <= 6):
                break
        # Bounded completion challenges heavier readings; a budget stop
        # is NOT an exclusion. Every accepted word has its syndrome checked.
        if best_word is not None and best_word.bit_count() > 6:
            singles, pairs = {}, {}
            incidence = [0]*len(rec["rows"])
            for v, col in enumerate(cols):
                singles.setdefault(col, 1 << v)
                for r in bits(col):
                    incidence[r] |= 1 << v
                for u in range(v):
                    pairs.setdefault(col ^ cols[u],
                                     (1 << v) | (1 << u))
            roots = list(range(0, width, q))
            local.shuffle(roots)
            priorities = [local.random() for _ in cols]
            cap = min(12, best_word.bit_count()-1)
            budget = min(24000, max(2400, 300*trials))
            all_variables = (1 << width)-1
            for root in roots:
                remaining = [max(1, budget//len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if not value:
                        assert syndrome(cols, word) == 0
                        observe(word)
                        return
                    left = min(cap, best_word.bit_count()-1)-word.bit_count()
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, best_word.bit_count()-1)-word.bit_count()
                    if left <= 2:
                        return
                    chosen, smallest = 0, width+1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        count = available.bit_count()
                        if not count:
                            return
                        if count < smallest:
                            chosen, smallest = available, count
                    candidates = list(bits(chosen))
                    candidates.sort(key=lambda v: (
                        -(cols[v] & value).bit_count(), priorities[v]))
                    live = allowed
                    for v in candidates:
                        bit = 1 << v
                        live &= ~bit
                        visit(value ^ cols[v], word | bit, live)
                        if remaining[0] <= 0:
                            break

                visit(cols[root], 1 << root,
                      all_variables & ~((1 << (root+1))-1))
                if best_word.bit_count() <= 6:
                    break
        assert best_word and syndrome(cols, best_word) == 0
        observe(best_word)
        rec["witnesses"] = sorted(
            retained, key=lambda w: (w.bit_count(), w))[:64]
        assert all(syndrome(cols, word) == 0
                   for word in rec["witnesses"])
        return best_word

    def component_pool(side):
        known, screened, signatures = [], [], set()
        mutable = [(0, j) for j in range(3)] if side == "A" else [
            (i, j) for i in range(2) for j in range(3, 7)
            if bcounts[j][i]]

        def loss(rec):
            penalty = 0
            for word in known:
                if syndrome(rec["cols"], word) == 0:
                    penalty += 5000*max(1, 9-word.bit_count())**2
            return rec["base"]+penalty

        def merit(rec):
            defect = len(rec["rows"])-rec["rank"]
            limit = 10 if side == "A" else 8
            preference = defect if 0 <= defect <= limit else -100
            # Rank preference follows the repeatedly measured bottleneck;
            # authoritative full k and actual logical witnesses decide later.
            return (rec["free4"], min(8, rec["reading"]), preference,
                    -loss(rec), min(12, rec["reading"]))

        def optimize(grid, steps):
            current = record(grid, side)
            best = current
            current_cost = best_cost = loss(current)
            for step in range(steps):
                positions = mutable
                active = [word for word in known
                          if syndrome(current["cols"], word) == 0]
                if active and step % 3 != 2:
                    nc = len(grid[0])
                    occupied = {v//q for v in bits(rng.choice(active))}
                    targeted = [p for p in mutable if p[1] in occupied
                                and p[1] < nc]
                    if targeted:
                        positions = targeted
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                count = len(old)
                kind = aplan[j] if side == "A" else (
                    j+i+model+cohort) % 3
                proposal = (term_variant(old, kind, side)
                            if step % 3 != 0 else None)
                grid[i][j] = (proposal if proposal is not None
                              else redraw_cell(old, kind, side))
                trial = record(grid, side)
                trial_cost = loss(trial)
                temperature = 60*(1-step/steps)**2+0.2
                accept = trial_cost <= current_cost or rng.random() < exp(
                    min(0.0, (current_cost-trial_cost)/temperature))
                if accept:
                    current, current_cost = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost
                else:
                    grid[i][j] = old
            return best

        pool = [optimize(fresh(side), 80)
                for _ in range(6 if side == "A" else 4)]
        side_offset = 100003*int(side == "B")
        for stage in range(3):
            pool.sort(key=loss)
            tested = []
            for rec in pool:
                signature = tuple(tuple(sorted(cell))
                                  for row in rec["grid"] for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                short = small_word(rec["cols"])
                rec["free4"] = short is None
                rec["rank"] = gf2_rank(rec["rows"])
                if side == "B":
                    assert all(
                        (functional & col).bit_count() % 2 == 0
                        for functional in b_functionals
                        for col in rec["cols"])
                    assert rec["rank"] <= 2*q-2
                rec["word"] = kernel_probe(
                    rec, side, 8,
                    diagnostic_seed+side_offset+1009*stage)
                rec["reading"] = rec["word"].bit_count()
                tested.append(rec)
                screened.append(rec)
                # Repair several actual witnesses; fixed skeleton words
                # cannot be removed by changing only the B tail.
                words = sorted(
                    set(rec["witnesses"]) | {rec["word"]},
                    key=lambda w: (w.bit_count(), w))[:6]
                for word in words:
                    movable = side == "A" or any(
                        v//q >= 3 for v in bits(word))
                    if (movable and word.bit_count() <= 10
                            and word not in known and len(known) < 64):
                        known.append(word)
            tested.sort(key=merit, reverse=True)
            if stage < 2:
                seeds = tested[:2] if tested else pool[:2]
                pool = [optimize(copy_grid(r["grid"]), 112) for r in seeds]
                pool += [optimize(fresh(side), 64) for _ in range(2)]
        if not screened:
            raise ValueError("empty A4 x C5 component portfolio")
        screened.sort(key=merit, reverse=True)
        finalists = screened[:5 if side == "A" else 3]
        for rec in finalists:
            for trials, offset in ((48,700001), (64,1700009)):
                word = kernel_probe(
                    rec, side, trials, diagnostic_seed+side_offset+offset)
                if word.bit_count() < rec["reading"]:
                    rec["word"], rec["reading"] = word, word.bit_count()
            rec["quality"] = (
                rec["free4"], rec["reading"], -loss(rec), rec["rank"])
        finalists.sort(key=merit, reverse=True)
        return finalists[:4 if side == "A" else 3]

    aa, bb = component_pool("A"), component_pool("B")
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        ab[0]["free4"] and ab[1]["free4"],
        min(ab[0]["reading"], ab[1]["reading"]),
        ab[0]["reading"]+ab[1]["reading"],
        ab[0]["quality"][2]+ab[1]["quality"][2]), reverse=True)

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Genuine logical representatives in the ORIGINAL quotient.

        No successful embedding returns n+1, not a distance certificate.
        Stabilizer reduction preserves commutation and nontriviality.
        """
        answer = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(
                set(rec.get("witnesses", ())) | {rec["word"]},
                key=lambda w: (w.bit_count(), w))[:4]
            candidates = set()
            for word in words:
                assert syndrome(rec["cols"], word) == 0
                for location in range(7 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((7*j+location)*q) for j in range(3))
                    else:
                        actual = word << (7*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            local = random.Random(
                diagnostic_seed+(3100009 if kind == "Z" else 4100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            reading = n+1
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best_word, best_weight = actual, actual.bit_count()
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            answer.append(reading)
        return tuple(answer)

    best_answer, best_quality = None, None
    for a, b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3,3,3]] and sb == planned_b
        assert all(sum(row) == 9 for row in sb)
        assert sorted(sum(sb[i][j] for i in range(2))
                      for j in range(7)) == [2,2,2,3,3,3,3]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert all(len({G.idx(g) for g in cell}) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        assert n == 1380 and k >= 600
        if not (425*n <= 1000*k < 475*n):
            continue
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        authoritative_a = [
            sum(((hx[t] >> (j*7*q)) & mask) << (j*q)
                for j in range(3)) for t in range(q)]
        authoritative_b = [
            row & ((1 << (7*q))-1) for row in hz[:2*q]]
        assert authoritative_a == a["rows"]
        assert authoritative_b == b["rows"]
        assert b["reading"] <= backbone_distance
        assert gf2_rank(b["rows"]) <= 2*q-2
        assert all(
            (functional & col).bit_count() % 2 == 0
            for functional in b_functionals for col in b["cols"])
        for j in range(3,7):
            assert sorted(
                quotient_phase[g]
                for i in range(2) for g in b["grid"][i][j]) == [0,1,2]
        for j, cell in enumerate(a["grid"][0]):
            assert cell[0] == ident
            assert [quotient_phase[g] for g in cell] == [0,1,2]
            ratio = central_ratio_profiles[composition][j]
            assert central_coordinate[cell[1]] != 0
            assert central_coordinate[cell[2]] == (
                ratio*central_coordinate[cell[1]]) % 5
        logicals = logical_readings(a, b, hx, hz, n)
        detected = min(8, a["reading"], b["reading"], *logicals)
        quality = (
            a["free4"] and b["free4"],
            k*detected*detected, detected,
            a["reading"]+b["reading"],
            -a["base"]-b["base"])
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_answer = {
                "idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "card": (composition, model, cohort),
                "component_readings": (a["reading"], b["reading"]),
                "component_kernel4_free": (a["free4"], b["free4"]),
                "component_ranks": (a["rank"], b["rank"]),
                "verified_localized_logical_readings": logicals,
                "backbone_cycle_weight": backbone_distance,
                "backbone_relative_orders": (
                    orders[ring_a], orders[ring_b],
                    orders[table[inverses[ring_a]][ring_b]]),
                "backbone_generated_subgroup_order": 20,
                "backbone_topology": "commuting normal-subgroup rings",
                "A_central_ratio_profile": central_ratio_profiles[composition],
                "verified_B_quotient_dependency_dimension": 2,
                "actual_B_check_defect": 2*q-b["rank"],
                "actual_A_check_defect": q-a["rank"],
                "exact_A_kernel6_free": a.get(
                    "exact_A_kernel6_free", False),
                "selection_reading_cap": 8,
                "pairing_quality": quality,
            }
    if best_answer is not None:
        return best_answer
    raise ValueError("A4 x C5 pairings missed the exact 0.450 rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete quotient-balanced rank/witness-selected cell."""
    cell = list(ctx["cells"][side][i][j])
    assert 0 <= len(cell) <= 5
    assert len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-saturating rolled supports on Z26:Z2, (2,6)x(2,4), did not suppress small logicals: all five measured codes had d=4, with abundant sector-1 witnesses.
# REFUTED: Degree>=3 columns and local syndrome-collision selection on Z17:Z2, (2,5)x(3,7), did not eliminate weight-10 sector-1 logical families: all five measured codes returned d=10 across many Acol/Bcol blocks.
# REFUTED: AGL(1,11) all-trinomial A and order-spectrum-screened B binomials/trinomials suppressed the sector-1 weight-six bottleneck; all five measured codes instead returned six, with thousands of X witnesses and k=660.
# REFUTED: Kernel-five exclusion alone removes the affine weight-six bottleneck; the all-even parent still measured five distance-six codes with thousands of sector-1 X and Z witnesses.
# REFUTED: Independent A5 spectrum/support optimization guarantees slot inequivalence; the measured parent produced 56 distinct codes from 60 slots, with four isomorphic duplicates.
# REFUTED: Longer-cycle A4 x C5 backbones and binding multi-witness logical repair produce five readings >=10 on (1,3)x(2,7); all five measured parent winners instead returned eight at k=615, with thousands of sector-one X and Z witnesses.
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
