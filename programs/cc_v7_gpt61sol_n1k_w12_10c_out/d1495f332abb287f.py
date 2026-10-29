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
CONCEPT = ("On (F9,+) semidirect C16, order 144, A=B=(1,3), invertible order-sixteen-supported cubic cells combined with independently witness-challenged full-generating cubics should produce five new measured readings >=14 and exceed the 0.400 incumbent's fitness 57.7. Failure of that five-code target refutes this selection hypothesis; readings remain provisional upper bounds.")
GROUP = ("Jump from cyclic C37 to the nonabelian order-144 group (F9,+) semidirect C16. Multiplication by 1+x, with x^2=-1 over F3, has exact order eight; C16 acts through it and has a central order-two kernel. The measured SL2(F5) and order-132 cubic beds reached fitness 57.6 and 57.7. These readings motivate another noncommutative cubic bed, not a distance-transfer theorem. Odd-augmentation elements of F2[C16] are units; actual verified block ranks and inverses enforce this fact.")
SHAPES = ("A=B=(1,3), q=144: n=1440, k>=576 and guaranteed rate 0.400; this shape allows q<=149. Equal components have 432 variables and 144 checks. The current parent's selected readings were eight, eight, ten, eight and ten, entirely in sector-1 Acol/Bcol blocks, and all measured k-kb values were zero. Invertible first blocks force full component rank; authoritative full ranks require k=576.")
FILLING = ("All six cells contain three distinct terms: both quantum check-weight sums are 9+3=12. Twenty cards cross four C16-unit allocations with five full-generating quotient profiles. Different cyclic subgroups are required when a component uses two units. Authoritative monomial lifts supply exact inverse columns and systematic kernel witnesses. Pair cancellation, budgeted anchored parity completion through twelve and randomized kernel descent challenge each component. Detected witnesses direct actual cell replacements, and independent finalist searches retain all earlier witnesses. Final native lifts verify assembly, ranks, CSS and weights. Neither component screens nor exhausted search budgets certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040_F9add_C16_13x13_unit16_witness12_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 144

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F9-additive semidirect C16 requires order 144")
    # F9 = F3[x]/(x^2+1); lambda=1+x has exact order eight.
    # (a,b,t) acts as z -> lambda^t*z + a+b*x.
    # The complement is C16, not C8: t=8 is a central involution.
    powers = ((1,0), (1,1), (0,2), (1,2),
              (2,0), (2,2), (0,1), (2,1))
    els = [(a,b,t) for t in range(16)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3, int(g[2]) % 16)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = powers[t % 8]
        return ((a+v*c-w*d) % 3,
                (b+v*d+w*c) % 3, (t+u) % 16)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % 16
        v,w = powers[ti % 8]
        return ((-v*a+w*b) % 3,
                (-v*b-w*a) % 3, ti)

    def idx(g):
        a,b,t = red(g)
        return 9*t+3*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _legacy_cyclic_ctx(rng, G, idx):
    """Retired C37 scorer; never called on the new nonabelian bed."""
    import itertools
    import math

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    card = idx % 20
    deck = card // 4
    mode = (card % 4 + idx // 20) % 4
    profiles = (
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
    )
    # Decks 3/4 additionally exclude doubled binomial gap relations.
    # No card is obtained by relabeling another card's final filling.
    profile_pairs = ((0, 0), (0, 1), (1, 1), (0, 0), (0, 1))
    pa, pb = profile_pairs[deck]
    mask = (1 << q) - 1

    def copy_cells(cells):
        return [[list(cell) for cell in row] for row in cells]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    def make_side(profile):
        # Voltage exclusions construct a four-cycle-free component.
        pattern = list(profiles[profile])
        rng.shuffle(pattern)
        if rng.randrange(2):
            pattern = [(b, a) for a, b in pattern]
        cells = [[[] for _ in pattern] for _ in range(2)]
        within = [set(), set()]
        cross = set()
        binomial_gaps = set()
        for j, counts in enumerate(pattern):
            for _ in range(1000):
                column = [rng.sample(G.elements, count) for count in counts]
                local = [set(), set()]
                legal = True
                for i in range(2):
                    for a, x in enumerate(column[i]):
                        for y in column[i][:a]:
                            d = G.mul(G.inv(x), y)
                            key = min(G.idx(d), G.idx(G.inv(d)))
                            # The parent only separated gaps within one
                            # row. Cross-row equality can collapse a
                            # binomial product in a mixed cofactor.
                            if (key in within[0] or key in within[1]
                                    or key in local[0] or key in local[1]):
                                legal = False
                            local[i].add(key)
                            if len(column[i]) == 2 and deck >= 3:
                                doubled = G.mul(d, d)
                                dk = min(G.idx(doubled),
                                         G.idx(G.inv(doubled)))
                                if dk in binomial_gaps:
                                    legal = False
                                for old_gap in binomial_gaps:
                                    old_element = G.elements[old_gap]
                                    twice = G.mul(old_element, old_element)
                                    tk = min(G.idx(twice),
                                             G.idx(G.inv(twice)))
                                    if tk == key:
                                        legal = False
                between = set()
                for x in column[0]:
                    for y in column[1]:
                        key = G.idx(G.mul(G.inv(x), y))
                        if key in cross or key in between:
                            legal = False
                        between.add(key)
                if legal:
                    for i in range(2):
                        cells[i][j] = column[i]
                        within[i].update(local[i])
                        if len(column[i]) == 2:
                            binomial_gaps.update(local[i])
                    cross.update(between)
                    break
            else:
                return None
        return cells

    def components(hx, hz):
        # Both components are extracted from the VERIFIED LP matrices.
        ar = []
        for i in range(ra):
            for t in range(q):
                source = hx[(i * cb) * q + t]
                row = 0
                for j in range(ca):
                    row |= ((source >> (j * cb * q)) & mask) << (j * q)
                ar.append(row)
        bmask = (1 << (cb * q)) - 1
        br = [hz[a * q + t] & bmask
              for a in range(rb) for t in range(q)]

        answer = []
        for rows, width in ((ar, ca * q), (br, cb * q)):
            cols = [0] * width
            for r, row in enumerate(rows):
                while row:
                    low = row & -row
                    cols[low.bit_length() - 1] |= 1 << r
                    row ^= low
            answer.append(cols)
        return answer

    def rotate(poly, shift):
        shift %= q
        if shift == 0:
            return poly
        return ((poly << shift) & mask) | (poly >> (q - shift))

    def multiply(a, b):
        """Cyclic F2 polynomial product, using verified column signatures."""
        answer = 0
        while b:
            low = b & -b
            answer ^= rotate(a, low.bit_length() - 1)
            b ^= low
        return answer

    def translate(word, shift):
        answer = 0
        for j in range(ca):
            answer |= rotate((word >> (j * q)) & mask, shift) << (j * q)
        return answer

    def cofactor_words(cols):
        """Generate and verify three-block component kernel words.

        At lift coordinate zero, each verified syndrome gives the two
        cyclic polynomials. Thus both left/right conventions are taken
        from lifted_matrices, not independently reconstructed.
        """
        polys = [(cols[j * q] & mask, cols[j * q] >> q)
                 for j in range(ca)]
        minors = {}
        for a, b in itertools.combinations(range(ca), 2):
            minors[a, b] = (
                multiply(polys[a][0], polys[b][1]) ^
                multiply(polys[b][0], polys[a][1]))
        words = []
        for a, b, c in itertools.combinations(range(ca), 3):
            word = (minors[b, c] << (a * q) |
                    minors[a, c] << (b * q) |
                    minors[a, b] << (c * q))
            if not word:
                continue
            syndrome = 0
            remaining = word
            while remaining:
                low = remaining & -remaining
                syndrome ^= cols[low.bit_length() - 1]
                remaining ^= low
            assert syndrome == 0, "cofactor disagrees with verified lift"
            words.append(word)
        return words

    def algebraic_penalty(cols):
        """Penalize witnessed component words below the target twelve."""
        minimum = len(cols)
        short = 0
        for word in cofactor_words(cols):
            weight = word.bit_count()
            minimum = min(minimum, weight)
            short += int(weight < 12)
            # Shifts d and -d yield translated copies of the same word.
            for shift in range(1, (q + 1) // 2):
                doubled = word ^ translate(word, shift)
                if doubled:
                    weight = doubled.bit_count()
                    minimum = min(minimum, weight)
                    short += int(weight < 12)
        return 25000 * max(0, 12 - minimum) ** 2 + 250 * short / q

    def statistics(cols):
        h = 2 * q
        adj = [0] * h
        incidence = [[] for _ in range(h)]
        four = 0
        for v, syndrome in enumerate(cols):
            checks = []
            while syndrome:
                low = syndrome & -syndrome
                u = low.bit_length() - 1
                checks.append(u)
                incidence[u].append(v)
                syndrome ^= low
            assert len(checks) == 3
            for a, u in enumerate(checks):
                for vcheck in checks[:a]:
                    if adj[u] & (1 << vcheck):
                        four += 1
                    adj[u] |= 1 << vcheck
                    adj[vcheck] |= 1 << u

        if four:
            return (four, 0, 0, 0), incidence

        triangle_sum = 0
        projected_four = 0
        local_triangles = []
        for u in range(h):
            local = 0
            for v in range(u + 1, h):
                common = (adj[u] & adj[v]).bit_count()
                projected_four += common * (common - 1) // 2
                if adj[u] & (1 << v):
                    triangle_sum += common
            neighbors = adj[u]
            while neighbors:
                low = neighbors & -neighbors
                v = low.bit_length() - 1
                local += (adj[u] & adj[v]).bit_count()
                neighbors ^= low
            # Remove triangles contributed by individual variables.
            local_triangles.append(local // 2 - len(incidence[u]))
        six = triangle_sum // 3 - len(cols)
        # This is a check-projection statistic, not an exact Tanner C8 count.
        projected_four //= 2
        return (0, six, projected_four, max(local_triangles)), incidence

    def objective(stats):
        four, six, projected, peak = stats
        if four:
            return 10000000 + 100000 * four
        if mode == 0:
            return float(six)
        if mode == 1:
            return six + projected / 32.0
        if mode == 2:
            return six + 4.0 * peak
        return six + projected / 8.0

    def inspect(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        cols = components(hx, hz)
        stats = [statistics(c)[0] for c in cols]
        cost = sum(objective(s) for s in stats)
        if not any(s[0] for s in stats):
            cost += sum(algebraic_penalty(c) for c in cols)
        return cost, stats, cols, hx, hz, n

    def tiny_kernel(cols):
        """Exact detection of component words of weight <=4."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return True
        pairs = set()
        for a, x in enumerate(cols):
            for y in cols[:a]:
                syndrome = x ^ y
                if syndrome in singles or syndrome in pairs:
                    return True
                pairs.add(syndrome)
        return False

    def weight_six(cols):
        """Exact weight-six screen for degree three with no four-cycles.

        Each variable in a weight-six word has at least three distinct
        adjacent variables in that word. There are two disjoint adjacent
        pairs, allowing a partition into two triples, each containing an
        adjacent pair. Enumerating those triples therefore suffices.
        Weight <=4 words must have been excluded first.
        """
        nc = len(cols)
        _, incidence = statistics(cols)
        adj = [set() for _ in cols]
        for variables in incidence:
            for v in variables:
                adj[v].update(variables)
                adj[v].discard(v)
        seen = {}
        for a in range(nc):
            for b in adj[a]:
                if b <= a:
                    continue
                pair = cols[a] ^ cols[b]
                for c in range(nc):
                    if c == a or c == b:
                        continue
                    lo, hi = min(a, c), max(b, c)
                    mid = a + b + c - lo - hi
                    tag = lo + nc * mid + nc * nc * hi
                    syndrome = pair ^ cols[c]
                    previous = seen.get(syndrome)
                    if previous is not None and previous != tag:
                        return True
                    seen[syndrome] = tag
        return False

    def elimination_reading(cols):
        """Verified component-kernel upper reading, never a lower bound."""
        words = cofactor_words(cols)
        best = min((word.bit_count() for word in words), default=len(cols))

        # Cofactors and binomial multiples alone need not contain the
        # minimum. Search cancellations between their translated orbits.
        translated = [
            [translate(word, shift) for shift in range(q)]
            for word in words
        ]
        for a, word in enumerate(words):
            for b in range(a, len(words)):
                for other in translated[b]:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())
        if best <= 10:
            return best

        order = list(range(len(cols)))
        for _ in range(40):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for v in order:
                syndrome, word = cols[v], 1 << v
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    best = min(best, word.bit_count())

            # Fundamental circuits can be long even when a cancellation
            # is short. Every tested XOR is an actual component word.
            light = sorted(basis, key=int.bit_count)[:40]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())

            # Also cancel fundamental circuits against algebraic words.
            for word in light[:12]:
                for orbit in translated:
                    for other in orbit:
                        candidate = word ^ other
                        if candidate:
                            best = min(best, candidate.bit_count())
            if best <= 10:
                return best
        return best

    keepers = []

    def remember(A, B, record):
        if any(s[0] for s in record[1]):
            return
        key = signature(A, B)
        if any(entry[1] == key for entry in keepers):
            return
        keepers.append((record[0], key, copy_cells(A), copy_cells(B), record))
        keepers.sort(key=lambda entry: entry[0])
        del keepers[8:]

    # Independent voltages, not automorphism or permutation orbit copies.
    starts = []
    for _ in range(12):
        A, B = make_side(pa), make_side(pb)
        if A is None or B is None:
            continue
        record = inspect(A, B)
        starts.append((record[0], A, B, record))
        remember(A, B, record)
    if not starts:
        raise ValueError("could not construct the degree-three profile")

    _, A, B, current = min(starts, key=lambda entry: entry[0])
    for step in range(240):
        grid = A if rng.randrange(2) == 0 else B
        positions = [(i, j) for i, row in enumerate(grid)
                     for j, cell in enumerate(row) if cell]
        i, j = rng.choice(positions)
        cell = grid[i][j]
        t = rng.randrange(len(cell))
        old = cell[t]
        new = G.pick(rng)
        while any(G._key(new) == G._key(g) for g in cell):
            new = G.pick(rng)
        cell[t] = new
        proposed = inspect(A, B)
        legal = not any(s[0] for s in proposed[1])
        temperature = q * (4.0 * (1.0 - step / 240.0) ** 2 + 0.05)
        accept = legal and (
            proposed[0] <= current[0] or
            rng.random() < math.exp((current[0] - proposed[0]) / temperature))
        if accept:
            current = proposed
            remember(A, B, current)
        else:
            cell[t] = old

    finalists = []
    for cost, key, A, B, record in keepers:
        cols = record[2]
        if any(tiny_kernel(c) for c in cols):
            continue
        readings = [elimination_reading(c) for c in cols]
        finalists.append((min(readings), sum(readings), -cost, A, B, record))
    finalists.sort(key=lambda entry: entry[:3], reverse=True)

    fallback = None
    selected = None
    for reading, total, negative_cost, A, B, record in finalists:
        _, stats, cols, hx, hz, n = record
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.375 <= k / n < 0.425):
            continue
        candidate = (A, B, record, k)
        if fallback is None:
            fallback = candidate
        if reading >= 12 and not any(weight_six(c) for c in cols):
            selected = candidate
            break

    # Preserve a valid slot if every finalist still contains a short word.
    if selected is None:
        selected = fallback
    if selected is None:
        for cost, key, A, B, record in keepers:
            hx, hz, n = record[3:]
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k > 0 and 0.375 <= k / n < 0.425:
                selected = (A, B, record, k)
                break
    if selected is None:
        raise ValueError("no optimized filling in rate cell 0.400")

    A, B, record, k = selected
    hx, hz, n = record[3:]
    assert css_ok(hx, hz)
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
            "screen_k": k, "cycle_cost": record[0]}


def NEW_CTX(rng, G, idx):
    """Twenty C16-unit/full-generating cards, using verified binary lifts."""
    q = G.q
    assert q == 144 and A_SHAPE == B_SHAPE == (1, 3)
    assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12 <= W_MAX
    width, mask = 3*q, (1 << q)-1
    card, cohort = idx % 20, idx // 20
    model, profile = card // 5, card % 5
    seed = rng.randrange(1 << 62)

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class IndexedGroup:
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

    probe = IndexedGroup()

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length()-1
            word ^= low

    def columns_of(rows, nc):
        columns = [0]*nc
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return columns

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # Small dummy batches obtain BOTH conventions from the scaffold.
    # Dummy zero partners are screening constructions, never submissions.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start+16)))
        cells = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, cells, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], cells)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    orders = []
    for g in range(q):
        h = ident
        for order in range(1, q+1):
            h = table[h][g]
            if h == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite-group order")

    def generates(a, b):
        reached, pending = {ident}, [ident]
        for g in pending:
            for h in (a, b):
                image = table[g][h]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    library, atom_keys = [], {}

    def add_atom(ids, kind=None, subgroup=None):
        ids = tuple(ids)
        assert len(set(ids)) == 3
        key = tuple(sorted(ids))
        previous = atom_keys.get(key)
        if previous is not None:
            return previous
        record = {"ids": ids, "kind": kind, "subgroup": subgroup}
        for side in ("A", "B"):
            rows = [
                blocks[side][ids[0]][r] ^ blocks[side][ids[1]][r]
                ^ blocks[side][ids[2]][r] for r in range(q)
            ]
            record[side] = rows
            record["cols"+side] = columns_of(rows, q)
            record["rank"+side] = gf2_rank(rows)
        atom_keys[key] = len(library)
        library.append(record)
        return len(library)-1

    # Distinct relative quotient occupancies, not whole-code orbit images.
    profiles = ((0, 0, 1), (0, 1, 2), (0, 1, 3),
                (0, 1, 5), (0, 1, 7))
    full_pools = [[] for _ in profiles]
    for p, coordinates in enumerate(profiles):
        seen = set()
        for attempt in range(1600):
            scale = rng.choice((1, 3, 5, 7, 9, 11, 13, 15))
            cell = [G.e]
            for t in coordinates[1:]:
                cell.append((rng.randrange(3), rng.randrange(3),
                             (scale*t) % 16))
            ids = tuple(G.idx(g) for g in cell)
            if len(set(ids)) != 3:
                continue
            if any(orders[table[inverses[a]][b]] == 2
                   for i, a in enumerate(ids) for b in ids[:i]):
                continue
            key = tuple(sorted(ids))
            if key in seen or not generates(ids[1], ids[2]):
                continue
            seen.add(key)
            full_pools[p].append(add_atom(ids))
            if len(full_pools[p]) >= 16:
                break
        if len(full_pools[p]) < 6:
            raise ValueError("full-generating quotient profile exhausted")

    unit_pools = [[], []]
    unit_seen = [set(), set()]
    patterns = ((0, 1, 2), (0, 1, 3))
    for g in range(q):
        if orders[g] != 16:
            continue
        powers = [ident]
        for _ in range(15):
            powers.append(table[powers[-1]][g])
        subgroup = frozenset(powers)
        assert len(subgroup) == 16
        for kind, pattern in enumerate(patterns):
            ids = tuple(powers[t] for t in pattern)
            # Remove root-normalization copies within this library.
            key = min(tuple(sorted(table[h][inverses[root]] for h in ids))
                      for root in ids)
            if key in unit_seen[kind]:
                continue
            t = add_atom(ids, kind, subgroup)
            if min(library[t]["rankA"], library[t]["rankB"]) != q:
                raise ValueError("odd C16 polynomial was not a verified unit")
            unit_seen[kind].add(key)
            unit_pools[kind].append(t)
    if min(map(len, unit_pools)) < 4:
        raise ValueError("insufficient C16 unit geometries")

    def inverse_columns(t, side):
        entry = library[t]
        name = "inverse"+side
        if name in entry:
            return entry[name]
        columns = entry["cols"+side]
        pivots = {}
        for c, source in enumerate(columns):
            value, word = source, 1 << c
            while value:
                p = value.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = value, word
                    break
                value ^= previous[0]
                word ^= previous[1]
        assert len(pivots) == q
        answer = []
        for r in range(q):
            value, word = 1 << r, 0
            while value:
                previous = pivots[value.bit_length()-1]
                value ^= previous[0]
                word ^= previous[1]
            assert syndrome(columns, word) == 1 << r
            answer.append(word)
        entry[name] = answer
        return answer

    def optimize(side):
        roles = ((0,), (1,), (0, 0), (0, 1))[model]
        count = len(roles)
        shift = 0 if side == "A" else 2+cohort % 3
        eligible = [unit_pools[kind] for kind in roles]
        for j in range(count, 3):
            eligible.append(full_pools[(profile+shift+j-count) % 5])
        local_seed = seed + (0 if side == "A" else 100003)
        seen, records = set(), []

        def note(record, word):
            if not word:
                return
            if word.bit_count() < record["reading"]:
                record["reading"], record["word"] = word.bit_count(), word
            if word.bit_count() <= 24:
                record["words"].add(word)

        def inspect(ids):
            ids = tuple(ids)
            key = tuple(sorted(tuple(sorted(library[t]["ids"])) for t in ids))
            if key in seen or len(set(ids)) != 3:
                return None
            if count == 2 and (
                    library[ids[0]]["subgroup"] ==
                    library[ids[1]]["subgroup"]):
                return None
            seen.add(key)
            rows = [
                sum(library[t][side][r] << (j*q)
                    for j, t in enumerate(ids)) for r in range(q)
            ]
            columns = columns_of(rows, width)
            if 0 in columns or len(set(columns)) != width:
                return None
            assert gf2_rank(rows) == q
            four = 0
            peak = 0
            for r, row in enumerate(rows):
                for other in rows[:r]:
                    overlap = (row & other).bit_count()
                    four += overlap*(overlap-1)//2
                    peak = max(peak, overlap)
            record = {
                "ids": ids, "rows": rows, "columns": columns,
                "reading": width+1, "word": None, "words": set(),
                "cycles": four, "peak": peak, "free4": False,
            }
            # [U,V,W], U invertible: complete systematic kernel basis.
            inverse = inverse_columns(ids[0], side)
            basis = []
            for c in range(q, width):
                word = 1 << c
                for r in bits(columns[c]):
                    word ^= inverse[r]
                basis.append(word)
                note(record, word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    candidate = word ^ other
                    if candidate.bit_count() <= record["reading"]:
                        note(record, candidate)
            record["words"] = set(sorted(
                record["words"], key=int.bit_count)[:32])
            assert record["word"] is not None
            assert syndrome(columns, record["word"]) == 0
            return record

        def completion(record, budget, turn):
            """Budgeted search: only discovered witnesses are conclusions."""
            columns = record["columns"]
            if "completion" not in record:
                singles, pairs = {}, {}
                incidence = [0]*q
                for c, value in enumerate(columns):
                    word = 1 << c
                    previous = singles.get(value)
                    if previous is not None:
                        note(record, previous ^ word)
                    else:
                        singles[value] = word
                    for r in bits(value):
                        incidence[r] |= word
                for i, value in enumerate(columns):
                    for j in range(i):
                        value2 = value ^ columns[j]
                        word = (1 << i) | (1 << j)
                        previous = singles.get(value2)
                        if previous is not None:
                            note(record, previous ^ word)
                        previous = pairs.get(value2)
                        if previous is not None:
                            note(record, previous ^ word)
                        else:
                            pairs[value2] = word
                record["completion"] = singles, pairs, incidence
                record["free4"] = record["reading"] > 4

            if record["reading"] <= 4 or budget <= 0:
                return
            singles, pairs, incidence = record["completion"]
            diagnostic = random.Random(local_seed+310001*turn)
            priorities = [diagnostic.random() for _ in columns]
            roots = [0, q, 2*q]
            diagnostic.shuffle(roots)
            all_variables = (1 << width)-1

            for root in roots:
                remaining = [max(1, budget//3)]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    left = min(12, record["reading"])-word.bit_count()
                    if not value:
                        note(record, word)
                        return
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        note(record, word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        note(record, word | pair)
                    left = min(12, record["reading"])-word.bit_count()
                    if left <= 2:
                        return
                    chosen, smallest = 0, width+1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        size = available.bit_count()
                        if not size:
                            return
                        if size < smallest:
                            chosen, smallest = available, size
                    choices = list(bits(chosen))
                    choices.sort(key=lambda c: (
                        -(columns[c] & value).bit_count(), priorities[c]))
                    live = allowed
                    for c in choices:
                        bit = 1 << c
                        live &= ~bit
                        visit(value ^ columns[c], word | bit, live)
                        if remaining[0] <= 0:
                            break

                # Regular component symmetry is transitive per block.
                # No corresponding full quantum-code symmetry is assumed.
                allowed = all_variables & ~((1 << (root+1))-1)
                visit(columns[root], 1 << root, allowed)

        def diagnose(record, trials, turn, budget):
            diagnostic = random.Random(local_seed+700001*turn)
            columns = record["columns"]
            retained = sorted(record["words"], key=int.bit_count)[:32]
            order = list(range(width))
            for _ in range(trials):
                diagnostic.shuffle(order)
                pivots, basis = {}, []
                for c in order:
                    value, word = columns[c], 1 << c
                    while value:
                        p = value.bit_length()-1
                        previous = pivots.get(p)
                        if previous is None:
                            pivots[p] = value, word
                            break
                        value ^= previous[0]
                        word ^= previous[1]
                    if not value:
                        basis.append(word)
                        note(record, word)
                light = sorted(set(basis+retained),
                               key=int.bit_count)[:20]
                for word in light:
                    for other in basis+retained:
                        note(record, word ^ other)
                diagnostic.shuffle(basis)
                for start in light[:6]:
                    word = start
                    for sweep in range(3):
                        changed = False
                        for other in basis+retained:
                            candidate = word ^ other
                            if candidate and (
                                    candidate.bit_count() < word.bit_count()):
                                word, changed = candidate, True
                        note(record, word)
                        if not changed:
                            break
                retained = sorted(record["words"],
                                  key=int.bit_count)[:32]
                record["words"] = set(retained)
                if record["reading"] <= 4:
                    break
            completion(record, budget, turn)
            record["words"] = set(sorted(
                record["words"], key=int.bit_count)[:32])
            assert syndrome(columns, record["word"]) == 0

        def merit(record):
            # Readings are capped to avoid chasing a single failed search.
            return (record["free4"], min(24, record["reading"]),
                    -sum(w.bit_count() <= 12 for w in record["words"]),
                    -record["cycles"], -record["peak"])

        for trial in range(72):
            record = inspect([rng.choice(pool) for pool in eligible])
            if record is not None:
                records.append(record)
        if not records:
            raise ValueError("no distinct unit/full-generating proposals")
        records.sort(key=lambda r: (
            -min(24, r["reading"]), r["cycles"], r["peak"]))

        keepers = []
        for number, record in enumerate(records[:10]):
            diagnose(record, 6, 1+number, 6000)
            keepers.append(record)
        current = max(keepers, key=merit)

        # Replace actual labels in blocks occupied by genuine witnesses.
        for step in range(16):
            occupied = sorted({c//q for c in bits(current["word"])})
            j = rng.choice(occupied) if occupied else rng.randrange(3)
            ids = list(current["ids"])
            choices = [t for t in eligible[j] if t != ids[j]]
            if not choices:
                continue
            ids[j] = rng.choice(choices)
            proposed = inspect(ids)
            if proposed is None:
                continue
            for word in current["words"]:
                if syndrome(proposed["columns"], word) == 0:
                    note(proposed, word)
            diagnose(proposed, 4, 31+step, 5000)
            if merit(proposed) >= merit(current):
                current = proposed
            elif (merit(proposed)[:2] == merit(current)[:2]
                  and rng.random() < 0.08):
                current = proposed
            keepers.append(proposed)
            keepers.sort(key=merit, reverse=True)
            del keepers[8:]

        finalists = sorted(keepers, key=merit, reverse=True)[:3]
        for number, record in enumerate(finalists):
            # Independent challenges only lower accumulated readings.
            diagnose(record, 32, 101+number, 30000)
            diagnose(record, 32, 211+number, 45000)
            record["cells"] = [[
                [G.elements[g] for g in library[t]["ids"]]
                for t in record["ids"]
            ]]
            record.pop("completion", None)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    aa, bb = optimize("A"), optimize("B")
    best = None
    for a in aa:
        for b in bb:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(probe, [
                [[G.idx(g) for g in cell] for cell in A[0]]
            ], [
                [[G.idx(g) for g in cell] for cell in B[0]]
            ])
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert n == 1440 and k == 576
            key = (
                a["free4"] and b["free4"],
                min(24, a["reading"], b["reading"]),
                min(24, a["reading"])+min(24, b["reading"]),
                -a["cycles"]-b["cycles"],
            )
            if best is None or key > best[0]:
                best = key, a, b, hx, hz, k

    key, a, b, cached_hx, cached_hz, k = best
    A, B = a["cells"], b["cells"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz
    assert n == 1440 and n-gf2_rank(hx)-gf2_rank(hz) == k == 576
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) == 12
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert sa == sb == [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    actual_a = [
        sum(((hx[r] >> (j*3*q)) & mask) << (j*q)
            for j in range(3)) for r in range(q)
    ]
    actual_b = [hz[r] & ((1 << width)-1) for r in range(q)]
    assert actual_a == a["rows"] and actual_b == b["rows"]
    return {
        "idx": idx, "cells": {"A": A, "B": B}, "screen_k": k,
        "card": (model, profile, cohort),
        "component_readings": (a["reading"], b["reading"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "kernel4_free": (a["free4"], b["free4"]),
        "unit_policy": ((0,), (1,), (0, 0), (0, 1))[model],
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected order-144 design cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal positive-cell rolling with uniform Z37 labels reliably avoids short sector-1 logicals; the parent instead has many d2s=2 slots and all five measured distances equal six, with minimum witnesses confined to sector 1.
# REFUTED: Cubic four-cycle-free lifting with weight-six screening eliminates the weight-ten bottleneck on this C37 bed; all five measured current-parent minima are ten, with thousands of verified sector-1 witnesses.
# REFUTED: Global-gap/cofactor selection produces five C37 readings >=12; the measured program instead returned eight, eight, ten, eight and ten, scored 31.36, and retained only twelve distinct codes among sixty slots.
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
