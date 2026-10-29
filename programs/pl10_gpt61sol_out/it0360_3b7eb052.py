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
CONCEPT = ("D17=C17 semidirect C2 with inversion, order 34, shapes (1,3)x(2,5): noncommuting degree-three components selected by balanced short-word exposure and common-noise decoder stress should improve the 0.350 frontier above fitness 5.0888.")
GROUP = ("Jump from C17 to nonabelian D17. No order-34 bed appears in BEDS. Its normal rotations have order seventeen, allowing long binomial differences; reflections permit noncommuting trinomials instead of the commuting single-row polynomial construction. These algebraic properties motivate the experiment but do not establish a circuit advantage.")
SHAPES = ("A=(1,3) and B=(2,5) give 102-variable/34-check and 170-variable/68-check components. The size factor seventeen permits q<=35; D17 fixes q=34, n=578, kb=204 and guaranteed rate6/17=0.352941. Complete ranks retain [0.325,0.375). Parent slot1 had two X failures collectively flipping62 logicals, while slots4 and7 also had rare failures. Enrichment involved several Acol/Bcol and Arow/Brow blocks, with too few failed shots to establish one persistent hotspot. Use unequal independently screened components rather than a claimed local repair.")
FILLING = ("A contains three mixed-coset noncommuting trinomials. B columns have degree three, with row totals eight/seven and order-seventeen doubletons. HX weighs twelve; HZ weighs eleven or ten; six-round CNOT count is25092. A deterministic canonical book of paired rotation-direction multisets removes obvious simultaneous automorphism and support-permutation copies before slot assignment. Fresh voltages are annealed using singleton actions obtained only from lifted_matrices. Full-rank connected components pass exact kernel<=4 exclusion; exact weight-five/six counts and balanced triple boundaries shortlist alternatives. Common-noise normalized-min-sum component tests and independent holdouts then prioritize the worse component. These omit ASC hooks, measurement noise and relay/LSD and are not circuit pL. Final verified CSS matrices enforce rate, connectivity and quantum logical<=4 exclusion. Target collective decoder ambiguity, not a larger distance upper bound.")
FAMILY_NAME = "r035_D17_q34_13x25_directioncontracts_decoderstress_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    # D17: x^17=y^2=1, y*x*y^-1=x^-1.
    # (a,b) encodes x^a*y^b.
    if not valid_q(q):
        raise ValueError("D17 requires order 34")
    els = [(a, b) for b in range(2) for a in range(17)]

    def red(g):
        return (int(g[0]) % 17, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c)) % 17, (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 17, 0) if b == 0 else (a, 1)

    def idx(g):
        a, b = red(g)
        return 17*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _shape_cards():
    """Canonical B count architectures, not row/column orbit copies.

    A column mask identifies its doubled cells. Mask seven is excluded,
    keeping column totals <=5. Row totals remain <=8. Canonicalization
    here concerns supports only, not complete Tanner-isomorphism testing.
    """
    from itertools import combinations_with_replacement, permutations

    row_permutations = list(permutations(range(3)))
    cards = set()
    for pattern in combinations_with_replacement(range(7), 5):
        extras = [sum((mask >> i) & 1 for mask in pattern)
                  for i in range(3)]
        if max(extras) > 3:
            continue
        augmentation = [
            sum((1 ^ ((mask >> i) & 1)) << j
                for j, mask in enumerate(pattern))
            for i in range(3)
        ]
        if gf2_rank(augmentation) != 3:
            continue
        canonical = min(
            tuple(sorted(
                sum(((mask >> p[i]) & 1) << i for i in range(3))
                for mask in pattern))
            for p in row_permutations
        )
        cards.add(canonical)

    def priority(pattern):
        extras = [sum((mask >> i) & 1 for mask in pattern)
                  for i in range(3)]
        return sum(extras), max(extras), tuple(sorted(extras)), pattern

    book = sorted(cards, key=priority)
    assert len(book) >= 20
    return book


def _retired_c17_ctx(rng, G, idx):
    """Inactive previous-bed search retained for provenance."""
    book = _shape_cards()
    design = idx % len(book)
    mode = idx // len(book)
    pattern = book[design]
    sa = [[1] * 5 for _ in range(3)]
    sb = [[1 + ((mask >> i) & 1) for mask in pattern]
          for i in range(3)]
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sa]) == 1
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sb]) == 3

    def grid(support):
        # Independent voltages, with distinct terms in every doubleton.
        return [[rng.sample(G.elements, count) for count in row]
                for row in support]

    q = G.q
    component_n = 5 * q
    block_mask = (1 << q) - 1

    def components(hx, hz):
        """Project the new components from the verified full lift."""
        ar = []
        for i in range(3):
            for r in range(q):
                row = hx[(i * 5) * q + r]
                packed = 0
                for j in range(5):
                    packed |= ((row >> (j * 5 * q)) &
                               block_mask) << (j * q)
                ar.append(packed)
        br = [hz[a * q + r] & ((1 << component_n) - 1)
              for a in range(3) for r in range(q)]
        return ar, br

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def connected(cols, nchecks):
        parent = list(range(nchecks))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for syndrome in cols:
            if not syndrome:
                return False
            first = syndrome & -syndrome
            a = first.bit_length() - 1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                syndrome ^= bit
        return len({root(r) for r in range(nchecks)}) == 1

    def four(rows):
        return sum(v * (v - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for v in [(row & other).bit_count()])

    def six(rows):
        """Exact Tanner six-cycles, including repeated intersections."""
        total = 0
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        triple = (common & rows[h]).bit_count()
                        total += (x * y * z -
                                  triple * (x + y + z) + 2 * triple)
        return total

    def boundary_screen(cols):
        """Exact kernel-word<=6 exclusion; triple-boundary heuristic."""
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return None
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return None
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return boundary

    def sampled_kernel(cols):
        """True kernel witnesses, but only an upper-bound search."""
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(40):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = syndrome, word
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome:
                    best = min(best, word.bit_count())
                    for other in basis:
                        best = min(best, (word ^ other).bit_count())
                    basis.append(word)
            if best < 16:
                break
        return best

    def short_logical(checks, stabilizers, n):
        """Exact non-stabilizer logical existence through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    for batch in range(3):
        pool = []
        for trial in range(160):
            A, B = grid(sa), grid(sb)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            fa, fb = four(ar), four(br)
            coarse = (max(fa, fb), fa + fb)
            pool.append((coarse, trial, A, B, hx, hz, n, ar, br))
        pool.sort(key=lambda item: (item[0], item[1]))

        finalists = []
        eligible = 0
        for item in pool:
            coarse, trial, A, B, hx, hz, n, ar, br = item
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # The augmentation argument motivates 72, but only these
            # full-CSS ranks decide whether a candidate is submitted.
            if not (n == 578 and k == 72 and 8 * k < n):
                continue
            ac, bc = columns(ar, component_n), columns(br, component_n)
            if not connected(ac, 3 * q) or not connected(bc, 3 * q):
                continue
            eligible += 1
            ba = boundary_screen(ac)
            bb = boundary_screen(bc) if ba is not None else None
            if ba is not None and bb is not None:
                da, db = sampled_kernel(ac), sampled_kernel(bc)
                sa, sb = six(ar), six(br)
                distance = min(da, db)
                # A sampled short witness is a warning, not a distance
                # certificate. Above twelve, prefer smaller boundaries
                # and cycle exposure rather than unbounded estimates.
                key = (int(distance < 12), max(ba, bb), ba + bb,
                       max(sa, sb), sa + sb, abs(da - db),
                       -min(distance, 16), coarse, trial)
                finalists.append((key, item, k))
            if eligible >= 12:
                break

        finalists.sort(key=lambda entry: entry[0])
        for key, item, k in finalists:
            _, _, A, B, hx, hz, n, _, _ = item
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "mode": mode,
                    "k": k, "screen": key}

    raise ValueError("no k=72 augmentation-rank-one/three filling passed screening")


def NEW_CTX(rng, G, idx):
    """Indexed D17 direction contracts and verified decoder-risk screening."""
    import math
    from itertools import combinations_with_replacement

    assert G.q == 34
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    q = G.q
    mask = (1 << q) - 1
    sa = [[3, 3, 3]]
    sb = [[1, 1, 2, 2, 2], [2, 2, 1, 1, 1]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert sum(sa[0]) + max(map(sum, zip(*sb))) == 12
    assert max(map(sum, sb)) + max(sa[0]) == 11

    def direction(value):
        value %= 17
        return min(value, 17-value)

    def canonical(a, b):
        # Intra-cell reversal, A-column permutations, and permutations
        # within each B count-column class. A common x->x^u action
        # must act on both sides, not independently on A and B.
        return min(
            (tuple(sorted(direction(u*d) for d in a)),
             tuple(sorted(direction(u*d) for d in b[:2])),
             tuple(sorted(direction(u*d) for d in b[2:])))
            for u in range(1, 9)
        )

    # Enumerated structural contracts, not saved-code orbit images.
    # Their equality patterns and cross-side direction ratios differ.
    cards = set()
    for a in combinations_with_replacement(range(1, 9), 3):
        if len(set(a)) < 2:
            continue
        for t in range(64):
            b = (
                1+t % 8,
                1+(t//8) % 8,
                1+(3*t+1) % 8,
                1+(5*t+t//8+2) % 8,
                1+(7*t+3*(t//8)+4) % 8,
            )
            if len(set(b)) < 3:
                continue
            cards.add(canonical(a, b))
    book = sorted(cards, key=lambda c: (
        len(set(c[0])), len(set(c[1]+c[2])), c))
    assert len(book) >= max(20, N_SLOTS)
    contract = book[idx * len(book) // N_SLOTS]
    ad, first_b, last_b = contract
    bd = first_b + last_b

    # Every action is obtained from the evaluator-convention helper.
    # No independent implementation of a left/right regular lift.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(cells, side):
        rows = []
        for cell_row in cells:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        out[r] ^= word << (j*q)
            rows.extend(out)
        return rows

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                return False
            bit = row & -row
            a = bit.bit_length()-1
            row ^= bit
            while row:
                bit = row & -row
                b = bit.bit_length()-1
                aa, bb = root(a), root(b)
                if aa != bb:
                    parent[bb] = aa
                row ^= bit
        return len({root(c) for c in range(width)}) == 1

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    def draw_cell(side, i, j):
        anchor = G.pick(rng)
        if side == "A":
            # One order-17 rotation difference and one reflection.
            # Nonidentity rotations cannot commute with reflections.
            d = ad[j]
            rotation = (d, 0)
            reflection = (rng.randrange(17), 1)
            assert G.mul(rotation, reflection) != G.mul(
                reflection, rotation)
            return [anchor, G.mul(anchor, rotation),
                    G.mul(anchor, reflection)]
        if sb[i][j] == 1:
            return [anchor]
        return [anchor, G.mul(anchor, (bd[j], 0))]

    def draw(side):
        support = sa if side == "A" else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(support)]

    def cycle_cost(rows, width):
        """Exact C4 and a cheap C6 proposal heuristic, not circuit pL."""
        nr = len(rows)
        adjacency = [0] * nr
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common*(common-1)//2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adjacency[i] & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j)-1)).bit_count()
                lower ^= bit
        # Columns all have degree three. Subtract variable stars.
        return four, max(0, triangles-width)

    def census(rows, width):
        """Exact <=4 exclusion, then weight-five/six component counts.

        After <=4 exclusion, equal pair/triple or triple/triple
        syndromes have disjoint supports. Each five- or six-word
        has ten corresponding unordered partitions.
        """
        cols = columns(rows, width)
        assert all(c.bit_count() == 3 for c in cols)
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        pairs = {}
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return None
                seen.add(value)
                pairs[value] = pairs.get(value, 0)+1

        triples = {}
        five = six = burden = 0
        loads = [0] * (width//q)
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    five += pairs.get(value, 0)
                    six += triples.get(value, 0)
                    triples[value] = triples.get(value, 0)+1
                    weight = value.bit_count()
                    if weight <= 4:
                        cost = (5-weight)**2
                        burden += cost
                        for block in {i//q, j//q, h//q}:
                            loads[block] += cost
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), burden

    def stress(rows, width, side, holdout=False):
        """Common-noise component min-sum, not ASC or relay/LSD.

        Decode a transmitted zero word on a synthetic BSC. A wrong
        valid codeword and nonconvergence both count as failures.
        Separate holdouts reduce selection noise; they do not turn
        this surrogate into a measured circuit failure rate.
        """
        import numpy as np

        nr = len(rows)
        maximum = max(row.bit_count() for row in rows)
        check_groups = []
        variable_groups = [[] for _ in range(width)]
        edge_variables = []
        locations = []
        matrix = np.zeros((nr, width), dtype=np.uint8)
        for r, row in enumerate(rows):
            group = []
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                edge = len(edge_variables)
                locations.append(r*maximum+len(group))
                group.append(edge)
                variable_groups[c].append(edge)
                edge_variables.append(c)
                matrix[r, c] = 1
                row ^= bit
            check_groups.append(group)

        ne = len(edge_variables)
        checks = np.full((nr, maximum), ne, dtype=np.int32)
        for r, group in enumerate(check_groups):
            checks[r, :len(group)] = group
        degree = np.array([len(g) for g in variable_groups])
        assert np.all(degree == 3)
        variables = np.full((width, int(degree.max())),
                            ne, dtype=np.int32)
        for c, group in enumerate(variable_groups):
            variables[c, :len(group)] = group
        ev = np.array(edge_variables, dtype=np.int32)
        locations = np.array(locations, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]

        shots = 192 if holdout else 96
        seed = (34035001 + 1009*idx + int(side == "B")
                + 1000003*int(holdout))
        channel = np.random.default_rng(seed)
        uniforms = channel.random((shots, width))
        outcomes = []
        for probability in (0.035, 0.060):
            errors = uniforms < probability
            magnitude = math.log((1-probability)/probability)
            prior = np.where(errors, -magnitude, magnitude)
            messages = prior[:, ev].copy()
            done = np.zeros(shots, dtype=bool)
            failed = np.zeros(shots, dtype=bool)
            late = 0
            for iteration in range(1, 33):
                padded = np.concatenate(
                    (messages, np.full((shots, 1), 1.0e6)), axis=1)
                incoming = padded[:, checks]
                absolute = np.abs(incoming)
                minimum_index = absolute.argmin(axis=2)
                minimum = absolute.min(axis=2)
                second = np.partition(absolute, 1, axis=2)[:, :, 1]
                signs = np.where(incoming >= 0, 1.0, -1.0)
                excluded = np.where(
                    positions == minimum_index[:, :, None],
                    second[:, :, None], minimum[:, :, None])
                outgoing = (0.8*signs.prod(axis=2)[:, :, None]
                            * signs * excluded)
                check_messages = outgoing.reshape(shots, -1)[:, locations]
                padded_checks = np.concatenate(
                    (check_messages, np.zeros((shots, 1))), axis=1)
                posterior = prior + padded_checks[:, variables].sum(axis=2)
                proposal = posterior[:, ev]-check_messages
                messages = np.clip(
                    0.25*messages+0.75*proposal, -24.0, 24.0)
                if iteration % 4 == 0:
                    hard = (posterior < 0).astype(np.uint8)
                    syndrome = (hard @ matrix.T) & 1
                    resolved = ~done & ~syndrome.any(axis=1)
                    failed[resolved] = hard[resolved].any(axis=1)
                    if iteration >= 16:
                        late += int(resolved.sum())
                    done |= resolved
                    if done.all():
                        break
            outcomes.append((int((failed | ~done).sum()),
                             int((~done).sum()), late))
        return (outcomes[0][0], outcomes[1][0],
                outcomes[0][1]+outcomes[1][1],
                outcomes[0][2]+outcomes[1][2])

    def portfolio(side):
        support = sa if side == "A" else sb
        width = len(support[0])*q
        sites = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
        pool = []
        signatures = set()
        serial = 0

        def assess(cells):
            rows = component(cells, side)
            deficit = len(rows)-gf2_rank(rows)
            four, six = cycle_cost(rows, width)
            cost = deficit, four, six
            energy = 80.0*deficit+(80.0*four+six/6.0)/q
            return energy, cost, rows

        def retain(cells, result):
            nonlocal serial
            _, cost, rows = result
            if cost[0] or (len(pool) >= 6 and cost > pool[-1][0]):
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in cells for cell in row)
            if signature in signatures:
                return
            signatures.add(signature)
            serial += 1
            pool.append((cost, serial, clone(cells), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(3):
            cells = draw(side)
            current = assess(cells)
            retain(cells, current)
            for step in range(180):
                i, j = rng.choice(sites)
                previous = cells[i][j]
                cells[i][j] = draw_cell(side, i, j)
                proposed = assess(cells)
                delta = proposed[0]-current[0]
                temperature = 10.0*(0.979**step)+0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                    if step % 12 == 0 or delta < 0:
                        retain(cells, current)
                else:
                    cells[i][j] = previous
            retain(cells, current)

        refined = []
        for cost, number, cells, rows in pool[:5]:
            if not connected(rows, width):
                continue
            exposure = census(rows, width)
            if exposure is None:
                continue
            risk = stress(rows, width, side)
            # Weight-five words are impossible for odd column degree,
            # but retain the exact count as an audited general screen.
            key = (int(exposure[0] > 0), risk, exposure, cost, number)
            refined.append((key, cells, rows, exposure, risk))
        refined.sort(key=lambda item: item[0])

        finalists = []
        for item in refined[:2]:
            _, cells, rows, exposure, initial = item
            holdout = stress(rows, width, side, True)
            combined = tuple(a+b for a, b in zip(initial, holdout))
            key = (int(exposure[0] > 0), combined, exposure,
                   cycle_cost(rows, width))
            finalists.append((key, cells, rows))
        finalists.sort(key=lambda item: item[0])
        return finalists

    def logical4(checks, stabilizers, n):
        """Exact non-stabilizer quantum logical existence through four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, value in enumerate(cols):
            word = 1 << i
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif nontrivial(word ^ old):
                return True
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    for batch in range(3):
        apool, bpool = portfolio("A"), portfolio("B")
        options = []
        for ak, A, expected_a in apool:
            for bk, B, expected_b in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 578 and 325*n <= 1000*k < 375*n):
                    continue
                # Verify cached action assembly against the full lift.
                actual_a = [
                    sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                        for j in range(3))
                    for r in range(q)
                ]
                actual_b = [
                    hz[a*q+r] & ((1 << (5*q))-1)
                    for a in range(2) for r in range(q)
                ]
                assert actual_a == expected_a
                assert actual_b == expected_b
                if not connected(hx+hz, n):
                    continue
                risk_a, risk_b = ak[1], bk[1]
                risk = tuple(v for a, b in zip(risk_a, risk_b)
                             for v in (max(a, b), a+b))
                ea, eb = ak[2], bk[2]
                exposure = tuple(v for a, b in zip(ea, eb)
                                 for v in (max(a, b), a+b))
                key = (risk, exposure, ak[3], bk[3], -k)
                options.append((key, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])
        for key, A, B, hx, hz, n, k in options:
            if logical4(hx, hz, n) or logical4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx) == 12
            assert max(row.bit_count() for row in hz) == 11
            assert 6*sum(row.bit_count() for row in hx+hz) == 25092
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            return {"idx": idx, "contract": contract, "k": k,
                    "screen": key, "cells": {"A": A, "B": B}}

    raise ValueError("no connected D17 target-rate pair passed verified screens")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without resampling or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: k=72 and d2s=12 guarantee zero-failure memory on the C17 (3,5)x(3,5) parent: slot1 had two X failures and62 collective logical flips, while slots0 and2 had zero failures in both bases.
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
