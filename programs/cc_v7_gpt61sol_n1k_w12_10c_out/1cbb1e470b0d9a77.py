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
CONCEPT = ("On cyclic Z37, order 37, A=(2,6), B=(2,6), fully split cubic components selected and repaired against verified weight-eight/ten kernel witnesses should raise all five measured distance readings from ten to twelve. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Frozen metacyclic Z37:Z1 with trivial action, q=37. This bed leads cell 0.400 at fitness 40.0, with five measured minima of ten. The shown global-gap and persistent-affine variants scored 31.36 and 37.12; short-cycle and gap constraints alone have not demonstrated improvement. Commutativity permits explicit cofactor witnesses checked against the scaffold's actual component syndromes.")
SHAPES = ("Frozen A=B=(2,6): n=1480, k>=592, guaranteed rate 0.400; q=37 is the largest admissible lift. Equal components have 222 variables and 74 checks. Parent minima were entirely sector-1 Acol/Bcol operators across both sides, with thousands of weight-ten witnesses and measured k-kb=0.")
FILLING = ("Only three (2,1) and three (1,2) columns per component: nine terms per row, three per column, hence w=12. Twenty cards prescribe unordered pairs of six inequivalent signed-gap spectra; accepted mutations preserve each binomial gap. Optimize actual cofactor words, binomial/trinomial multiples and translated pair cancellations. Independently select components using bounded connected-subset orbit collisions and randomized kernel circuits, then repair surviving short witnesses by changing labels in their occupied blocks. Verify actual syndromes, full ranks, CSS, weights and the 0.400 rate band. Truncated subset enumeration and sampled readings do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040_m37r1s1_26x26_split3_orbitwitness_repair_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # metacyclic Z_37 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 37, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty gap-spectrum cards with actual short-witness repair."""
    import itertools
    import math

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    card = idx % 20
    mode = (card % 4 + idx // 20) % 4
    diagnostic_seed = rng.randrange(1 << 62)
    profiles = ([(2, 1)] * 3 + [(1, 2)] * 3,)
    mask = (1 << q) - 1

    # A common group automorphism, inversion of individual gaps and
    # permutation cannot turn one prescribed six-gap set into another.
    # This is not a complete Tanner-isomorphism test.
    gap_sets, gap_keys = [], set()
    for gaps in itertools.combinations(range(1, (q + 1) // 2), 6):
        key = min(tuple(sorted(min((u*d) % q, (-u*d) % q)
                               for d in gaps))
                  for u in range(1, q))
        if key not in gap_keys:
            gap_keys.add(key)
            gap_sets.append(gaps)
        if len(gap_sets) == 6:
            break
    assert len(gap_sets) == 6
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    pa, pb = plans[card]

    def copy_cells(cells):
        return [[list(cell) for cell in row] for row in cells]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    def make_side(profile):
        pattern = list(profiles[0])
        rng.shuffle(pattern)
        scale = rng.randrange(1, q)
        gaps = [(scale*d) % q for d in gap_sets[profile]]
        rng.shuffle(gaps)
        cells = [[[] for _ in pattern] for _ in range(2)]
        cross = set()
        for j, counts in enumerate(pattern):
            gap = gaps[j]
            for _ in range(1000):
                column = []
                for count in counts:
                    origin = rng.randrange(q)
                    column.append(
                        [(origin, 0), ((origin + gap) % q, 0)]
                        if count == 2 else [(origin, 0)])
                between = {
                    G.idx(G.mul(G.inv(x), y))
                    for x in column[0] for y in column[1]
                }
                if len(between) != 2 or between & cross:
                    continue
                for i in range(2):
                    cells[i][j] = column[i]
                cross.update(between)
                break
            else:
                return None
        return cells

    def mutate_cell(cell):
        # Translating the entire binomial preserves its signed gap;
        # its voltage relative to the other cell in the column changes.
        offset = G.pick(rng)
        while G._key(offset) == G._key(G.e):
            offset = G.pick(rng)
        return [G.mul(offset, g) for g in cell]

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

    def rotate(poly, shift):
        shift %= q
        if not shift:
            return poly
        return ((poly << shift) & mask) | (poly >> (q - shift))

    def translate(word, shift):
        return sum(rotate((word >> (j*q)) & mask, shift) << (j*q)
                   for j in range(ca))

    def translate_syndrome(syndrome, shift):
        return (rotate(syndrome & mask, shift) |
                (rotate(syndrome >> q, shift) << q))

    def multiply(a, b):
        result = 0
        while b:
            bit = b & -b
            result ^= rotate(a, bit.bit_length() - 1)
            b ^= bit
        return result

    def syndrome_of(word, cols):
        syndrome = 0
        while word:
            bit = word & -word
            syndrome ^= cols[bit.bit_length() - 1]
            word ^= bit
        return syndrome

    def cofactor_words(cols):
        # Polynomial coordinates come from VERIFIED lift columns.
        polys = [(cols[j*q] & mask, cols[j*q] >> q)
                 for j in range(ca)]
        assert all(
            cols[j*q+t] ==
            (rotate(a, t) | (rotate(b, t) << q))
            for j, (a, b) in enumerate(polys) for t in range(q))
        minors = {}
        for a, b in itertools.combinations(range(ca), 2):
            minors[a, b] = (
                multiply(polys[a][0], polys[b][1]) ^
                multiply(polys[b][0], polys[a][1]))
        words = []
        for a, b, c in itertools.combinations(range(ca), 3):
            word = (minors[b, c] << (a*q) |
                    minors[a, c] << (b*q) |
                    minors[a, b] << (c*q))
            if word:
                assert syndrome_of(word, cols) == 0
                words.append(word)
        return words

    assessment_cache = {}

    def assess(cols):
        """Actual algebraic upper readings and short-witness multiplicities."""
        key = tuple(cols)
        old = assessment_cache.get(key)
        if old is not None:
            return old
        stats = statistics(cols)[0]
        if stats[0]:
            answer = (objective(stats), 4, 1000000, [], [])
        else:
            words = cofactor_words(cols)
            orbits = [[translate(word, t) for t in range(q)]
                      for word in words]
            reading = len(cols)
            short = 0

            def record(word):
                nonlocal reading, short
                if word:
                    weight = word.bit_count()
                    reading = min(reading, weight)
                    short += int(weight < 12)

            for word, orbit in zip(words, orbits):
                record(word)
                for t in range(1, (q + 1) // 2):
                    record(word ^ orbit[t])
                # These are genuine three-term scalar multiples, not
                # independently reconstructed component matrices.
                for s in range(1, q):
                    pair = word ^ orbit[s]
                    for t in range(s + 1, q):
                        record(pair ^ orbit[t])
            for a, word in enumerate(words):
                for orbit in orbits[a:]:
                    for other in orbit:
                        record(word ^ other)

            cost = (50000 * max(0, 12 - reading) ** 2 +
                    4 * short + objective(stats) / q)
            answer = (cost, reading, short,
                      [w for orbit in orbits for w in orbit], words)
        assessment_cache[key] = answer
        if len(assessment_cache) > 96:
            del assessment_cache[next(iter(assessment_cache))]
        return answer

    def inspect(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        cols = components(hx, hz)
        stats = [statistics(c)[0] for c in cols]
        cost = sum(assess(c)[0] for c in cols)
        return cost, stats, cols, hx, hz, n

    def component(side, grid):
        """Extract a standalone component using the verified constructor."""
        width = ca * q
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        cols = [0] * width
        for r, row in enumerate(rows):
            row &= (1 << width) - 1
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return cols

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

    def canonical_word(word, syndrome=0):
        # Every translate can be anchored at a variable in the first
        # occupied protograph block. This normalizes only a VALID
        # cyclic symmetry of the verified component.
        first = ((word & -word).bit_length() - 1) // q
        roots = (word >> (first*q)) & mask
        best = None
        while roots:
            bit = roots & -roots
            shift = -(bit.bit_length() - 1)
            candidate = translate(word, shift)
            if best is None or candidate < best[0]:
                best = (candidate, translate_syndrome(syndrome, shift))
            roots ^= bit
        return best

    def elimination_reading(cols):
        """Bounded orbit collisions and actual randomized kernel circuits.

        Connected-subset beams are deliberately truncated. Failure to
        find a short word is NOT an exact exclusion or distance certificate.
        """
        algebraic = assess(cols)
        best = algebraic[1]
        witnesses = {}

        def observe(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= 10 and len(witnesses) < 64:
                assert syndrome_of(word, cols) == 0
                normal, _ = canonical_word(word)
                witnesses[normal] = weight

        width = len(cols)
        incidence = [0] * (2*q)
        for v, syndrome in enumerate(cols):
            value = syndrome
            while value:
                bit = value & -value
                incidence[bit.bit_length() - 1] |= 1 << v
                value ^= bit
        adjacency = [0] * width
        for variables in incidence:
            value = variables
            while value:
                bit = value & -value
                adjacency[bit.bit_length() - 1] |= variables ^ bit
                value ^= bit

        seen = {}

        def orbit_collision(word, syndrome):
            # Normalize the syndrome orbit and apply the SAME
            # translation to its associated variable subset.
            roots = syndrome & mask
            if not roots:
                roots = syndrome >> q
            if not roots:
                observe(word)
                return
            normal = None
            while roots:
                bit = roots & -roots
                shift = -(bit.bit_length() - 1)
                value = translate_syndrome(syndrome, shift)
                if normal is None or value < normal[0]:
                    normal = (value, shift)
                roots ^= bit
            key, shift = normal
            shifted = translate(word, shift)
            previous = seen.get(key)
            if previous is not None:
                observe(previous ^ shifted)
            else:
                seen[key] = shifted

        states = {}
        for j in range(ca):
            word = 1 << (j*q)
            states[word] = cols[j*q]
            orbit_collision(word, cols[j*q])

        caps = (300, 1800, 3600, 9000 + 1000*mode)
        for cap in caps:
            following = {}
            ordered = sorted(states.items(),
                             key=lambda item: (item[1].bit_count(), item[0]))
            stopped = False
            for word, syndrome in ordered:
                boundary = 0
                remaining = word
                while remaining:
                    bit = remaining & -remaining
                    boundary |= adjacency[bit.bit_length() - 1]
                    remaining ^= bit
                boundary &= ~word
                while boundary:
                    bit = boundary & -boundary
                    v = bit.bit_length() - 1
                    candidate, value = canonical_word(
                        word | bit, syndrome ^ cols[v])
                    boundary ^= bit
                    if candidate in following:
                        continue
                    following[candidate] = value
                    orbit_collision(candidate, value)
                    if len(following) >= cap:
                        stopped = True
                        break
                if stopped:
                    break
            states = following
            if not states:
                break

        diagnostic = random.Random(diagnostic_seed)
        order = list(range(width))
        retained = sorted(set(algebraic[3]), key=int.bit_count)[:24]
        for trial in range(32):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
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
                    observe(word)
            light = sorted(basis, key=int.bit_count)[:20]
            candidates = []
            for word in light:
                for other in basis + retained:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= 16:
                        candidates.append(candidate)
            diagnostic.shuffle(basis)
            for word in light[:4]:
                for _ in range(2):
                    changed = False
                    for other in basis + retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    observe(word)
                    if not changed:
                        break
                candidates.append(word)
            retained = sorted(set(retained + candidates),
                              key=int.bit_count)[:32]
            if best <= 6:
                break
        return best, len(witnesses), list(witnesses)

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

    # Independent voltages with prescribed gap spectra, not orbit copies.
    starts = []
    for _ in range(16):
        A, B = make_side(pa), make_side(pb)
        if A is None or B is None:
            continue
        record = inspect(A, B)
        starts.append((record[0], A, B, record))
        remember(A, B, record)
    if not starts:
        raise ValueError("could not construct the degree-three profile")

    _, A, B, current = min(starts, key=lambda entry: entry[0])
    steps = 320
    for step in range(steps):
        grid = A if rng.randrange(2) == 0 else B
        i, j = rng.randrange(2), rng.randrange(6)
        old = grid[i][j]
        grid[i][j] = mutate_cell(old)
        proposed = inspect(A, B)
        legal = not any(s[0] for s in proposed[1])
        temperature = 4.0 * (1.0 - step / steps) ** 2 + 0.03
        accept = legal and (
            proposed[0] <= current[0] or
            rng.random() < math.exp((current[0] - proposed[0]) / temperature))
        if accept:
            current = proposed
            remember(A, B, current)
        else:
            grid[i][j] = old

    def grid_key(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    pools = [[], []]
    for side_number, side in enumerate(("A", "B")):
        unique = {}
        for cost, key, saved_A, saved_B, record in keepers:
            grid = (saved_A, saved_B)[side_number]
            cols = record[2][side_number]
            unique.setdefault(grid_key(grid), (grid, cols))
        candidates = sorted(unique.values(), key=lambda item: assess(item[1])[0])

        def entry_of(grid, cols):
            reading, hits, witnesses = elimination_reading(cols)
            if tiny_kernel(cols):
                reading = min(reading, 4)
            quality = (reading, -hits, -assess(cols)[0])
            return (quality, copy_cells(grid), cols, witnesses)

        for grid, cols in candidates[:3]:
            pools[side_number].append(entry_of(grid, cols))
        if not pools[side_number]:
            raise ValueError("empty independently screened component pool")
        pools[side_number].sort(key=lambda item: item[0], reverse=True)

        # Cutting-plane-style repair: old witnesses are tested against
        # each proposed VERIFIED component before a fresh search.
        current_entry = pools[side_number][0]
        bank = list(current_entry[3])
        for repair_round in range(4):
            if current_entry[0][0] >= 12 or not bank:
                break
            proposals = []
            for _ in range(6):
                grid = copy_cells(current_entry[1])
                witness = rng.choice(bank)
                blocks = [j for j in range(ca)
                          if (witness >> (j*q)) & mask]
                j = rng.choice(blocks)
                i = rng.randrange(2)
                grid[i][j] = mutate_cell(grid[i][j])
                cols = component(side, grid)
                if statistics(cols)[0][0]:
                    continue
                algebraic = assess(cols)
                survivors = sum(syndrome_of(word, cols) == 0
                                for word in bank)
                key = (algebraic[1], -survivors,
                       -algebraic[2], -algebraic[0])
                proposals.append((key, grid, cols))
            if not proposals:
                break
            _, grid, cols = max(proposals, key=lambda item: item[0])
            repaired = entry_of(grid, cols)
            pools[side_number].append(repaired)
            # Explore a repaired filling even if its new search discovers
            # another short family; retain the best alternatives separately.
            current_entry = repaired
            bank = list(dict.fromkeys(bank + repaired[3]))[-96:]
        pools[side_number].sort(key=lambda item: item[0], reverse=True)
        pools[side_number] = pools[side_number][:3]

    proposals = []
    for a in pools[0]:
        for b in pools[1]:
            quality = (min(a[0][0], b[0][0]),
                       a[0][0] + b[0][0],
                       a[0][1] + b[0][1],
                       a[0][2] + b[0][2])
            proposals.append((quality, a, b))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for quality, a, b in proposals:
        A, B = a[1], b[1]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1480 and k > 0 and
                375*n <= 1000*k < 425*n):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
                "screen_k": k, "quality": quality,
                "gap_spectra": (gap_sets[pa], gap_sets[pb]),
                "component_readings": (a[0][0], b[0][0]),
                "short_witness_orbits": (-a[0][1], -b[0][1])}
    raise ValueError("no repaired component pair in rate cell 0.400")


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal positive-cell rolling with uniform Z37 labels reliably avoids short sector-1 logicals; the parent instead has many d2s=2 slots and all five measured distances equal six, with minimum witnesses confined to sector 1.
# REFUTED: Balanced cubic four-cycle-free components and component weight-six screening remove the weight-ten bottleneck on this Z37 bed; all five measured current-parent codes instead returned ten, with thousands of sector-1 witnesses and k-kb=0.
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
