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
CONCEPT = ("Cyclic C17 in the frozen metacyclic encoding, order 17, shapes (3,5)x(3,5): degree-three components with augmentation ranks one and three should improve worse-basis circuit fitness above 7.2396 at verified k=72 and 24480 six-round CNOTs.")
GROUP = ("Frozen byte-for-byte under L1. This exact C17 bed produced two measured codes with zero failures in both bases, supporting a filling experiment. Order 17 is already the largest lift permitted by size factor 34; no group advantage is inferred from other beds.")
SHAPES = ("Frozen equal (3,5) components each have 85 variables and 51 checks; n=578, kb=68 and guaranteed rate 2/17=0.117647. Require full verified k=72, rate 0.124567 in cell 0.100. Component ranks 49 and 51 give dimensions 36 and 34. Parent slots 1,4,7 had only one or two failures per affected basis, enriching several Acol/Bcol and Arow/Brow blocks; these samples do not identify a unique repair block.")
FILLING = ("A remains fifteen monomials, with gauge-normalized independent cycle voltages and indexed minor contracts. B redistributes exactly fifteen terms into canonical support cards with column totals three, row totals 4..7 and augmentation rank three; zero cells and multi-term cells replace the parent's added doubletons. HX weight is eight, HZ weight at most ten, and extraction uses 24480 CNOTs versus 26112..27744 in the measured parent. Anneal verified component four/six cycles, then screen exact ranks, connectivity and kernel words<=6. Checked cyclic syndrome orbits count component weight-eight words and quartet-boundary exposure; capped sampled witnesses penalize short-word concentration. Full verified CSS ranks and logical<=4 tests decide submission. Target rare collective failures such as slot 1's two X failures flipping 62 logical qubits, not a presumed distance-to-pL equivalence.")
FAMILY_NAME = "r010_m17r1s1_35x35_aug13_degree3_quartet_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same metacyclic rule; factor 34 requires q<=17.
    m, r, s = 17, 1, 1
    if q != m * r:
        raise ValueError("this metacyclic bed requires order 17")
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1] % r, m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Optimize indexed fifteen-term contracts using verified components."""
    from itertools import combinations_with_replacement, permutations

    # Each column has degree three, but its terms need not occupy
    # three different protograph rows. Quotient support permutations,
    # rather than spending slots on those images.
    types = [(a, b, 3 - a - b)
             for a in range(4) for b in range(4 - a)]
    row_permutations = list(permutations(range(3)))
    cards = set()
    for pattern in combinations_with_replacement(types, 5):
        totals = tuple(sum(column[i] for column in pattern)
                       for i in range(3))
        if min(totals) < 4 or max(totals) > 7:
            continue
        augmentation = [
            sum((column[i] & 1) << j
                for j, column in enumerate(pattern))
            for i in range(3)]
        if gf2_rank(augmentation) != 3:
            continue
        # Require connected support before searching lift voltages.
        links = [{i} for i in range(3)]
        for column in pattern:
            active = {i for i, count in enumerate(column) if count}
            for i in active:
                links[i].update(active)
        reached = {0}
        for _ in range(3):
            reached.update(set().union(*(links[i] for i in reached)))
        if len(reached) != 3:
            continue
        canonical = min(
            tuple(sorted(tuple(column[p[i]] for i in range(3))
                         for column in pattern))
            for p in row_permutations)
        cards.add(canonical)

    def priority(pattern):
        totals = [sum(column[i] for column in pattern)
                  for i in range(3)]
        return (max(totals) - min(totals),
                sum(count == 3 for column in pattern for count in column),
                max(totals), pattern)

    book = sorted(cards, key=priority)
    assert len(book) >= 20
    design = idx % len(book)
    mode = idx // len(book)
    if mode >= 3:
        raise ValueError("degree-three support/minor contract book exhausted")
    pattern = book[design]
    minor = (2, 3, 4)[mode]
    sa = [[1] * 5 for _ in range(3)]
    sb = [[column[i] for column in pattern] for i in range(3)]
    assert sum(map(sum, sa)) == sum(map(sum, sb)) == 15
    assert all(sum(column) == 3 for column in pattern)
    assert _grid_weights(sa, sb) <= min(10, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sa]) == 1
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sb]) == 3

    def grid(support, side):
        if side == "B":
            return [[rng.sample(G.elements, count) for count in row]
                    for row in support]
        # Remove monomial row/column gauges. The two fixed minor
        # voltages leave six independent inter-cell cycle voltages.
        result = [[[(0, 0)] for _ in range(5)] for _ in range(3)]
        result[1][1] = [(1, 0)]
        result[2][1] = [(minor, 0)]
        for i in (1, 2):
            for j in range(2, 5):
                result[i][j] = [G.pick(rng)]
        return result

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

    def sampled_kernel(cols, side):
        """Sample verified kernel words and translation-orbit concentration.

        Minima remain upper bounds; multiplicities are sampled, not
        exact spectra. Every finalist uses the same permutation stream.
        """
        srng = random.Random(170035 + 1009 * idx + int(side == "B"))
        order = list(range(len(cols)))
        best = len(cols) + 1
        short = set()

        def translate(word, shift):
            return sum(
                ((((word >> (j * q)) & block_mask) << shift)
                 & block_mask
                 | (((word >> (j * q)) & block_mask) >> (q - shift)))
                << (j * q) for j in range(5))

        def offer(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight > 14:
                return
            first = next(j for j in range(5)
                         if (word >> (j * q)) & block_mask)
            part = (word >> (first * q)) & block_mask
            images = []
            while part:
                bit = part & -part
                shift = (-bit.bit_length() + 1) % q
                images.append(translate(word, shift))
                part ^= bit
            short.add(min(images))

        for _ in range(48):
            srng.shuffle(order)
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
                if not syndrome and word:
                    basis.append(word)
                    offer(word)

            # Basis XORs preserve kernel membership. Shorten the basis
            # before counting sampled low-weight orbit representatives.
            basis.sort(key=int.bit_count)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposal = word ^ other
                    offer(proposal)
                    proposed_weight = proposal.bit_count()
                    if proposal and proposed_weight < weight:
                        word, weight = proposal, proposed_weight
                basis[i] = word
                offer(word)
            if best <= 8:
                break

        vulnerable = [word for word in short if word.bit_count() <= 12]
        loads = [sum(bool((word >> (j * q)) & block_mask)
                     for word in vulnerable) for j in range(5)]
        return best, len(vulnerable), max(loads, default=0)

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

    def component(side, cells):
        """Extract only from the scaffold's verified zero-partner lift."""
        if side == "A":
            rows = lifted_matrices(G, cells, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], cells)[1]
        mask = (1 << component_n) - 1
        return [row & mask for row in rows]

    def orbit_quartets(cols):
        """Exact weight-eight count after exclusion of kernels <=6.

        Four-support translation orbits have size seventeen. Equal
        canonical syndrome orbits give q pair collisions, and every
        weight-eight word has 35 unordered four/four partitions.
        Small quartet boundaries are a separate BP-risk surrogate.
        """
        assert q == 17 and all(col.bit_count() == 3 for col in cols)

        def rotate(value, shift):
            return ((value << shift) & block_mask) | (value >> (q - shift))

        def translated(value, shift, blocks):
            return sum(rotate((value >> (r * q)) & block_mask, shift)
                       << (r * q) for r in range(blocks))

        for j in range(5):
            seed = cols[j * q]
            assert all(cols[j * q + t] == translated(seed, t, 3)
                       for t in range(q)), "verified syndrome action mismatch"

        bits = [1 << i for i in range(component_n)]
        least_shift = {}
        histogram = {}
        collisions = total = 0
        loads = [0] * 5
        for anchor in range(0, component_n, q):
            first_block = anchor // q
            for a in range(anchor + 1, component_n - 2):
                pair = cols[anchor] ^ cols[a]
                for b in range(a + 1, component_n - 1):
                    triple = pair ^ cols[b]
                    for c in range(b + 1, component_n):
                        vertices = (anchor, a, b, c)
                        word = bits[anchor] | bits[a] | bits[b] | bits[c]
                        canonical = True
                        for v in vertices[1:]:
                            if v // q != first_block:
                                break
                            shift = v % q
                            image = sum(bits[(u // q) * q
                                             + (u % q - shift) % q]
                                        for u in vertices)
                            if image < word:
                                canonical = False
                                break
                        if not canonical:
                            continue

                        syndrome = triple ^ cols[c]
                        weight = syndrome.bit_count()
                        if weight <= 4:
                            penalty = (5 - weight) ** 2
                            total += penalty
                            for block in {v // q for v in vertices}:
                                loads[block] += penalty

                        shift = None
                        for r in range(2, -1, -1):
                            part = (syndrome >> (r * q)) & block_mask
                            if part not in (0, block_mask):
                                if part not in least_shift:
                                    least_shift[part] = min(
                                        range(q), key=lambda t: rotate(part, t))
                                shift = least_shift[part]
                                break
                        # Degree-three quartet syndromes have weight
                        # at most twelve, so a nonzero invariant block
                        # of weight seventeen is impossible.
                        assert shift is not None
                        key = translated(syndrome, shift, 3)
                        old = histogram.get(key, 0)
                        collisions += old
                        histogram[key] = old + 1

        assert collisions % 35 == 0
        return q * (collisions // 35), max(loads), total

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    def portfolio(side, support):
        """Cycle-voltage search preserving the indexed count/minor card."""
        import math

        sites = ([(i, j) for i in (1, 2) for j in range(2, 5)]
                 if side == "A" else
                 [(i, j) for i in range(3) for j in range(5)
                  if support[i][j]])
        pool = []
        seen = set()
        serial = 0

        def assess(cells):
            rows = component(side, cells)
            c4, c6 = four(rows), six(rows)
            return rows, (c4, c6), (80 * c4 + c6 / 8) / q

        def remember(cells, rows, cycles):
            nonlocal serial
            if len(pool) >= 8 and cycles > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in cells for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((cycles, serial, clone(cells), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(3):
            cells = grid(support, side)
            rows, cycles, energy = assess(cells)
            remember(cells, rows, cycles)
            for step in range(180):
                proposal = clone(cells)
                i, j = rng.choice(sites)
                proposal[i][j] = rng.sample(G.elements, support[i][j])
                pr, pc, pe = assess(proposal)
                temperature = 5.0 * (0.982 ** step) + 0.05
                delta = pe - energy
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    cells, rows, cycles, energy = proposal, pr, pc, pe
                    remember(cells, rows, cycles)

        eligible = []
        target_rank = 49 if side == "A" else 51
        for cycles, serial, cells, rows in pool:
            if gf2_rank(rows) != target_rank:
                continue
            cols = columns(rows, component_n)
            if not connected(cols, 3 * q):
                continue
            boundary = boundary_screen(cols)
            if boundary is None:
                continue
            sampled = sampled_kernel(cols, side)
            preliminary = (int(sampled[0] < 12),
                           boundary, cycles, sampled[1:], serial)
            eligible.append((preliminary, cells, cols, sampled, cycles))
        eligible.sort(key=lambda item: item[0])

        finalists = []
        for preliminary, cells, cols, sampled, cycles in eligible[:4]:
            eight, peak, exposure = orbit_quartets(cols)
            # Exact short ambiguity and small-boundary exposure precede
            # optimistic sampled-distance estimates. None is measured pL.
            key = (int(sampled[0] < 12), eight, peak, exposure,
                   sampled[1], sampled[2], cycles[0], cycles[1],
                   -min(sampled[0], 14))
            finalists.append((key, cells))
        finalists.sort(key=lambda item: item[0])
        return finalists

    for batch in range(3):
        apool = portfolio("A", sa)
        bpool = portfolio("B", sb)
        combinations = []
        for akey, A in apool:
            for bkey, B in bpool:
                # Balance the worse component before total exposure.
                key = tuple(value for t in range(8)
                            for value in (max(akey[t], bkey[t]),
                                          akey[t] + bkey[t]))
                key += (akey[8], bkey[8])
                combinations.append((key, A, B))
        combinations.sort(key=lambda item: item[0])

        for key, A, B in combinations:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 578 and k == 72 and 8 * k < n):
                continue
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(10, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == 24480
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "mode": mode, "minor": minor,
                    "support_card": pattern, "k": k, "screen": key}

    raise ValueError("no degree-three k=72 filling passed verified screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell, including prescribed zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: k=72,w=9,d2s=12 certifies zero-failure C17 memory: slots 0 and 1 shared these readings, yet slot 1 had one Z and two X failures, s=7.078, versus zero failures and s=7.370 in slot 0.
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
