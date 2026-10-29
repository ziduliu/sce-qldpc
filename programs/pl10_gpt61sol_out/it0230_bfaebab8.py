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
CONCEPT = ("For C7 semidirect C6 with action x->x^2, order 42, A=(1,2), B=(2,6), four-cycle-free A fourtons with odd central-parity counts and compensated B row weights should raise cell-0.300 circuit fitness above 6.1594 by reducing the parent's collective Z-basis failures.")
GROUP = ("Keep the exact validated group, order and encoding byte-for-byte under L1. This centre-two order-42 bed leads the 0.300 frontier at 6.1594. Its measured slot variation motivates a filling experiment, not a claim that a different group would perform better.")
SHAPES = ("Frozen A=(1,2), B=(2,6) have 84 variables/42 checks and 252 variables/84 checks. At q=42, n=588, kb=168 and guaranteed rate 2/7 target cell 0.300; complete verified ranks retain [0.275,0.325). Parent failures were diffuse across Acol/Bcol sector-one blocks, usually with 1.0-1.2x enrichment, and affected nearly every logical. No persistent Acol/Brow hotspot is established. Retain these unequal components while reallocating their term budgets.")
FILLING = ("Twenty contracts cross five disjoint canonical A-geometry panels with four B order schedules; extra slots advance to new panels. A weights become (4,4), B row totals (8,8), with two degree-two matching columns and four degree-three columns. Thus HX weighs 10 or 11, HZ twelve, and extraction uses 28224 rather than 25704 six-round CNOTs. Each A fourton has an odd number of odd-C2 quotient labels; verified column syndromes confirm a check combination equal to all ones, excluding odd component kernels. Verified lifts also confirm zero A four-cycles. Select A by short-kernel exclusion and anchored weight-eight ambiguity; optimize B long matching holonomies, connectivity, short kernels and boundary exposure. Final verified matrices screen actual rate, CSS and quantum logicals through four. Target slot8's 100/68250 Z versus 51/200000 X failures and widespread collective flips, not the distance upper bound. The additional gate exposure is an explicit risk to be measured.")
FAMILY_NAME = "r030_C7sdC6_action2_12x26_parityfourton_compensatedB_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C6 requires q=42")
    # x^7=y^6=1, y*x*y^-1=x^2.
    # The action has order three; y^3 is central of order two.
    els = [(a, b) for b in range(6) for a in range(7)]
    action = (1, 2, 4, 1, 2, 4)

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _f21_columns(hx, hz, q):
    """Project component columns from the VERIFIED lifted matrices.

    A: sector 1 of X checks at B-column zero.
    B: sector 1 of Z checks at A-column zero.
    No independent left/right-lift implementation is used.
    """
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    ac = [0] * (ca * q)
    bc = [0] * (cb * q)
    amask = sum(((1 << q) - 1) << (j * cb * q)
                for j in range(ca))
    bmask = (1 << (cb * q)) - 1
    for i in range(ra):
        for r in range(q):
            bits = hx[(i * cb) * q + r] & amask
            check = 1 << (i * q + r)
            while bits:
                low = bits & -bits
                c = low.bit_length() - 1
                ac[(c // (cb * q)) * q + c % q] |= check
                bits ^= low
    for a in range(rb):
        for r in range(q):
            bits = hz[a * q + r] & bmask
            check = 1 << (a * q + r)
            while bits:
                low = bits & -bits
                bc[low.bit_length() - 1] |= check
                bits ^= low
    return ac, bc


def _f21_stats(cols, nchecks):
    """Exact short-word counts, Tanner four-cycles and connectivity."""
    parent = list(range(nchecks))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checkpairs = {}
    c4 = 0
    for col in cols:
        rows = []
        bits = col
        while bits:
            low = bits & -bits
            rows.append(low.bit_length() - 1)
            bits ^= low
        if rows:
            for r in rows[1:]:
                parent[root(r)] = root(rows[0])
        for t, r in enumerate(rows):
            for s in rows[t + 1:]:
                pair = (r, s)
                old = checkpairs.get(pair, 0)
                c4 += old
                checkpairs[pair] = old + 1

    colset = set(cols)
    repeated = len(cols) - len(colset)
    pairseen = {}
    words3 = words4 = 0
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            syndrome = x ^ y
            words3 += int(syndrome in colset)
            old = pairseen.get(syndrome, 0)
            words4 += old
            pairseen[syndrome] = old + 1
    components = len({root(r) for r in range(nchecks)})
    return (repeated, words3, words4), c4, components


def _f21_traps(cols):
    """Small boundaries of three-variable sets; repeated triple syndromes.

    The first statistic is a trapping-set heuristic. The second also
    detects weight-six component words when shorter words are absent.
    Neither statistic is asserted to predict circuit pL exactly.
    """
    boundary = repeated = 0
    seen = {}
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def _legacy_C19_CTX(rng, G, idx):
    """Inactive historical C19 search; never called by the new builder."""
    q = G.q
    profiles = (
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 1)),
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 2)),
        ((0, 3, 1, 2, 1), (3, 0, 2, 1, 2)),
    )
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    architecture = (idx // 5) % len(pairs)
    pa, pb = pairs[architecture]
    sa, sb = profiles[pa], profiles[pb]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(sum(row) for row in sa) + max(
        sum(row[j] for row in sb) for j in range(5)) <= 12
    assert max(sum(row) for row in sb) + max(
        sum(row[j] for row in sa) for j in range(5)) <= 12

    def normalized(values):
        if not values:
            return ()
        return min(tuple(sorted((v - origin) % q for v in values))
                   for origin in values)

    def geometry_key(grids):
        """Remove obvious automorphism, cell-gauge and permutation copies.

        This is an internal-label invariant, not a complete classifier
        of arbitrary Tanner-graph isomorphisms.
        """
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                columns = [
                    tuple(normalized([u * v % q for v in grid[i][j]])
                          for i in range(2))
                    for j in range(5)
                ]
                direct = tuple(sorted(columns))
                swapped = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(direct, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    geometry_rng = random.Random(190031 + 1009 * architecture)
    geometry_number = idx % 5 + 5 * (idx // 20)
    book = []
    seen = set()
    for proposal in range(20000):
        grids = []
        for supports in (sa, sb):
            sites = [(i, j) for i in range(2) for j in range(5)
                     if supports[i][j] == 2]
            # Distinct undirected binomial differences prevent repeated
            # internal binomial directions within a component.
            directions = geometry_rng.sample(range(1, (q + 1) // 2),
                                               len(sites))
            steps = dict(zip(sites, directions))
            grid = []
            for i in range(2):
                row = []
                for j in range(5):
                    count = supports[i][j]
                    if count == 0:
                        cell = ()
                    elif count == 1:
                        cell = (0,)
                    elif count == 2:
                        cell = (0, steps[i, j])
                    else:
                        cell = (0,) + tuple(
                            geometry_rng.sample(range(1, q), count - 1))
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        key = geometry_key(grids)
        if key in seen:
            continue
        seen.add(key)
        book.append(grids)
        if len(book) > geometry_number:
            break
    if len(book) <= geometry_number:
        raise ValueError("insufficient fresh internal-label geometries")
    ga, gb = book[geometry_number]

    def draw(geometry):
        # These independent cell offsets change inter-cell holonomies.
        # No candidate is obtained by relabeling a saved complete code.
        grid = []
        for row in geometry:
            out = []
            for values in row:
                offset = rng.randrange(q)
                out.append([((v + offset) % q,) for v in values])
            grid.append(out)
        return grid

    def kernel_upper(cols, trials=64):
        """Verified kernel witnesses from randomized fundamental circuits.

        Combining two sampled basis words also gives genuine kernel
        words. A large result is not a lower-bound distance certificate.
        """
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            basis = []
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
                if syndrome == 0 and word:
                    best = min(best, word.bit_count())
                    for previous in basis:
                        best = min(best, (word ^ previous).bit_count())
                    basis.append(word)
            if best <= 6:
                break
        return best

    def low4(checks, stabilizers, n):
        """Exact quantum-logical existence test through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        bits = [1 << c for c in range(n)]
        seen = {0: 0}
        for c, syndrome in enumerate(columns):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = bits[c]
            elif nontrivial(bits[c] ^ old):
                return True
        for c in range(n):
            for d in range(c):
                syndrome = columns[c] ^ columns[d]
                word = bits[c] ^ bits[d]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def small_boundary(cols):
        """Exact existence screen through weight six plus triple boundaries.

        If different supports of size <=3 have equal syndromes, their
        symmetric difference is a nonzero component kernel word <=6.
        A passing result is exact only within this short-word window.
        """
        seen = {0}
        for value in cols:
            if value in seen:
                return True, 0
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return True, 0
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True, boundary
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return False, boundary

    def anchored_exposure(cols):
        """Translation-anchored weight-seven/eight collision exposure.

        Called only after exclusion of component words through six.
        Equal syndromes of distinct four-subsets then have disjoint
        supports and witness weight eight. Four/three collisions witness
        weight seven. Counts are anchored exposures, not exact spectra
        and not a model of ASC ordering or circuit-level pL.
        """
        size = len(cols)
        anchors = tuple(range(0, size, q))
        anchor_set = set(anchors)
        anchored = {}
        for a in anchors:
            smaller = {h for h in anchors if h < a}
            allowed = [h for h in range(size)
                       if h != a and h not in smaller]
            length = len(allowed)
            for bi in range(length - 2):
                b = allowed[bi]
                ab = cols[a] ^ cols[b]
                for ci in range(bi + 1, length - 1):
                    c = allowed[ci]
                    abc = ab ^ cols[c]
                    for di in range(ci + 1, length):
                        d = allowed[di]
                        syndrome = abc ^ cols[d]
                        anchored[syndrome] = anchored.get(syndrome, 0) + 1

        lookup = anchored.get
        seven = eight = 0
        for a in range(size - 2):
            x = cols[a]
            for b in range(a + 1, size - 1):
                ab = x ^ cols[b]
                for c in range(b + 1, size):
                    seven += lookup(ab ^ cols[c], 0)

        for a in range(size - 3):
            x = cols[a]
            for b in range(a + 1, size - 2):
                ab = x ^ cols[b]
                for c in range(b + 1, size - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, size):
                        eight += lookup(abc ^ cols[d], 0)
        # Each anchored support was compared once with itself.
        eight -= sum(anchored.values())
        assert eight >= 0
        return seven, eight

    def component_rank(cols):
        rows = [0] * (2 * q)
        for c, syndrome in enumerate(cols):
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length() - 1] |= 1 << c
                syndrome ^= bit
        return gf2_rank(rows)

    for batch in range(2):
        shortlist = []
        for trial in range(96):
            A, B = draw(ga), draw(gb)
            hx, hz, n = lifted_matrices(G, A, B)
            ac, bc = _f21_columns(hx, hz, q)
            ast = _f21_stats(ac, 2 * q)
            bst = _f21_stats(bc, 2 * q)
            if ast[2] != 1 or bst[2] != 1:
                continue
            if component_rank(ac) != 2 * q:
                continue
            if component_rank(bc) != 2 * q:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 551 or k != 171:
                continue
            bad = max(ast[0], bst[0])
            totals = tuple(x + y for x, y in zip(ast[0], bst[0]))
            cycles = (max(ast[1], bst[1]), ast[1] + bst[1])
            base = (bad, totals, cycles)
            item = (base, trial, k, A, B, ac, bc, hx, hz, n)
            shortlist.append(item)
            shortlist.sort(key=lambda entry: (entry[0], entry[1]))
            del shortlist[8:]

        finalists = []
        for item in shortlist:
            base, trial, k, A, B, ac, bc, hx, hz, n = item
            bad_a, ba = small_boundary(ac)
            bad_b, bb = small_boundary(bc)
            da, db = kernel_upper(ac, 40), kernel_upper(bc, 40)
            distance = min(da, db)
            key = (int(bad_a) + int(bad_b),
                   int(distance < 8), max(ba, bb), ba + bb,
                   base[2], -min(distance, 10), abs(da - db), trial)
            finalists.append((key, item, not (bad_a or bad_b)))
        finalists.sort(key=lambda entry: entry[0])

        refined = []
        # Refine two alternatives, not an expensive full-spectrum search
        # on every proposal. Bad short-word candidates remain fallback
        # designs, but never outrank a passing short-word candidate.
        for key, item, safe in finalists[:2]:
            _, trial, _, _, _, ac, bc, _, _, _ = item
            if safe:
                ea, eb = anchored_exposure(ac), anchored_exposure(bc)
                exposure = (max(ea[0], eb[0]), ea[0] + eb[0],
                            max(ea[1], eb[1]), ea[1] + eb[1])
            else:
                exposure = (10**12,) * 4
            refined_key = (key[0], key[1], exposure, key[2:], trial)
            refined.append((refined_key, item))
        refined.sort(key=lambda entry: entry[0])

        for key, item in refined:
            _, _, k, A, B, _, _, hx, hz, n = item
            if low4(hx, hz, n) or low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "architecture": architecture,
                    "geometry": geometry_number, "k": k,
                    "screen": key, "cells": {"A": A, "B": B}}

    raise ValueError("No connected full-rank C19 design passed screening")


def _q42_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= tag
            row ^= bit
    return cols


def _q42_component(G, side, grid):
    """Extract only the scaffold's verified component action."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    width = len(grid[0])*G.q
    mask = (1 << width)-1
    return [row & mask for row in rows], width


def _q42_cycles(rows, cols):
    """Exact component Tanner four- and six-cycle counts."""
    from itertools import combinations

    adjacency = [dict() for _ in rows]
    neighborhoods = []
    for syndrome in cols:
        ns = []
        while syndrome:
            bit = syndrome & -syndrome
            ns.append(bit.bit_length()-1)
            syndrome ^= bit
        neighborhoods.append(ns)
        for a, b in combinations(ns, 2):
            adjacency[a][b] = adjacency[a].get(b, 0)+1
            adjacency[b][a] = adjacency[b].get(a, 0)+1

    four = sum(v*(v-1)//2
               for a, ns in enumerate(adjacency)
               for b, v in ns.items() if a < b)
    six = 0
    for a, ns in enumerate(adjacency):
        for b, ab in ns.items():
            if b <= a:
                continue
            for c, bc in adjacency[b].items():
                if c > b and c in ns:
                    six += ab*bc*ns[c]
    # Inclusion-exclusion removes triangles reusing a variable.
    for ns in neighborhoods:
        for a, b, c in combinations(ns, 3):
            six -= (adjacency[a][b]+adjacency[a][c]
                    +adjacency[b][c]-2)
    assert six >= 0
    return four, six


def _q42_small_kernel(cols, through_six):
    """Exact component exclusion through four, or six on the A side."""
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return True
        seen.add(syndrome)
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            if value in seen:
                return True
            seen.add(value)
    if through_six:
        for i, syndrome in enumerate(cols):
            for j in range(i):
                pair = syndrome ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True
                    seen.add(value)
    return False


def _q42_sample(rng, cols, q, trials):
    """Actual verified kernel witnesses, not lower bounds on distance."""
    order = list(range(len(cols)))
    minimum = len(cols)+1
    short = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
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
            if not syndrome:
                basis.append(word)

        # Basis replacements preserve genuine kernel membership.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10 and len(short) < 8192:
                    short.add(word)
                for other in basis[:i]:
                    candidate = word ^ other
                    if not candidate:
                        continue
                    cw = candidate.bit_count()
                    minimum = min(minimum, cw)
                    if cw <= 10 and len(short) < 8192:
                        short.add(candidate)
                    if cw < weight:
                        word, weight = candidate, cw
                basis[i] = word
        if minimum <= 4:
            break

    mask = (1 << q)-1
    vulnerable = [word for word in short if word.bit_count() <= 8]
    loads = [sum(bool((word >> (j*q)) & mask) for word in vulnerable)
             for j in range(len(cols)//q)]
    return minimum, len(vulnerable), max(loads, default=0)


def _q42_low4(checks, stabilizers, n):
    """Exact quantum-logical existence test through weight four."""
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

    cols = _q42_columns(checks, n)
    bits = [1 << c for c in range(n)]
    seen = {0: 0}
    for c, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[c]
        elif nontrivial(bits[c] ^ old):
            return True
    for c, syndrome in enumerate(cols):
        for d in range(c):
            value = syndrome ^ cols[d]
            word = bits[c] ^ bits[d]
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif nontrivial(word ^ old):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Parity-fourton panels and compensated, long-holonomy B components."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 42
    a_card, policy = divmod(idx, 4)
    sa = [[4, 4]]
    sb = [[1, 1, 2, 2, 1, 1],
          [1, 1, 1, 1, 2, 2]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert sum(sa[0]) + max(map(sum, zip(*sb))) == 11
    assert max(map(sum, sb)) + max(sa[0]) == 12

    elements = list(G.elements)
    orders = {}
    inverses = {g: G.inv(g) for g in elements}
    for g in elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        assert g in orders
    palettes = {t: [g for g in elements if orders[g] == t]
                for t in (7, 14, 6)}
    assert all(palettes.values())

    # Cache only actions extracted from the verified construction.
    # Their XOR composition uses linearity, not another lift convention.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][g] = tuple(row & mask for row in hx)
        actions["B"][g] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        rows = []
        for grid_row in grid:
            block = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        block[r] ^= bits << (j * q)
            rows.extend(block)
        width = len(grid[0]) * q
        return rows, _q42_columns(rows, width)

    def analyze(side, grid):
        rows, cols = component(side, grid)
        four, six = _q42_cycles(rows, cols)
        short, _, parts = _f21_stats(cols, len(rows))
        deficit = len(rows) - gf2_rank(rows)
        if side == "A":
            assert all(c.bit_count() == 4 for c in cols)
            parity_checks = sum(1 << r for r, g in enumerate(elements)
                                if g[1] & 1)
            # This identity is checked on the helper-derived syndromes.
            # It excludes odd-weight A kernels despite even column degree.
            assert all((c & parity_checks).bit_count() % 2 == 1
                       for c in cols)
        else:
            assert all(c.bit_count() == (2 if j < 2*q else 3)
                       for j, c in enumerate(cols))
        energy = (100 * (parts - 1) + 50 * deficit +
                  (100 * four + six / 8.0) / q +
                  50 * sum(short) / q)
        coarse = (parts - 1, int(any(short)), deficit, four, six)
        return energy, coarse, rows, cols

    def a_identity(grid):
        """Remove explicit A gauges, automorphisms and column exchange.

        This is not a complete arbitrary-Tanner isomorphism classifier.
        Automorphisms are x->x^u, y->x^c*y, with u!=0 modulo seven.
        """
        images = []
        sums = (0, 1, 3, 0, 1, 3)
        for u in range(1, 7):
            for c in range(7):
                cells = []
                for cell in grid[0]:
                    image = [((u*a + c*sums[b]) % 7, b)
                             for a, b in cell]
                    normal = min(
                        tuple(sorted(G.idx(G.mul(g, G.inv(origin)))
                                     for g in image))
                        for origin in image)
                    cells.append(normal)
                images.append(tuple(sorted(cells)))
        return min(images)

    def fourton(grng):
        for _ in range(1024):
            cell = [G.e] + grng.sample(
                [g for g in elements if g != G.e], 3)
            if sum(g[1] & 1 for g in cell) % 2 != 1:
                continue
            if any(orders[G.mul(g, inverses[h])] < 3
                   for g, h in combinations(cell, 2)):
                continue
            if not any(G.mul(g, h) != G.mul(h, g)
                       for g, h in combinations(cell, 2)):
                continue
            return cell
        raise ValueError("parity-fourton palette exhausted")

    # Twelve genuinely different A geometries per indexed panel.
    # The deterministic book is shared conceptually, not mutable state.
    # Panels never deliberately reuse an A gauge/automorphism class.
    grng = random.Random(4244881)
    start, stop = 12 * a_card, 12 * (a_card + 1)
    book, identities = [], set()
    accepted = 0
    for attempt in range(200000):
        A = [[fourton(grng), fourton(grng)]]
        differences = set()
        repeated = False
        for cell in A[0]:
            for g in cell:
                for h in cell:
                    if g == h:
                        continue
                    value = G.mul(g, inverses[h])
                    if value in differences:
                        repeated = True
                        break
                    differences.add(value)
                if repeated:
                    break
            if repeated:
                break
        if repeated:
            continue

        # The algebraic prefilter is never accepted as the lift screen.
        rows, cols = component("A", A)
        four, _ = _q42_cycles(rows, cols)
        if four:
            continue
        identity = a_identity(A)
        if identity in identities:
            continue
        identities.add(identity)
        if accepted >= start:
            book.append(A)
        accepted += 1
        if accepted >= stop:
            break
    if len(book) != 12:
        raise ValueError("distinct four-cycle-free A panel exhausted")

    # The two matching columns alone form cycles of length 2*order.
    # The remaining doubletons vary independently within each schedule.
    matching_order = (14, 14, 7, 14)[policy]
    binomial_orders = (
        (14, 14, 14, 14),
        (7, 7, 7, 7),
        (14, 6, 14, 6),
        (6, 6, 6, 6),
    )[policy]
    double_sites = ((0, 2), (0, 3), (1, 4), (1, 5))
    desired_orders = dict(zip(double_sites, binomial_orders))

    def b_cell(i, j):
        if j == 0 or (j == 1 and i == 0):
            return [G.e]
        if j == 1:
            return [rng.choice(palettes[matching_order])]
        anchor = G.pick(rng)
        if sb[i][j] == 1:
            return [anchor]
        relative = rng.choice(palettes[desired_orders[i, j]])
        return [anchor, G.mul(anchor, relative)]

    def draw_B():
        return [[b_cell(i, j) for j in range(6)] for i in range(2)]

    def anchored_boundary(cols):
        """Worst-block triple-boundary exposure; a structural surrogate."""
        burdens = []
        for anchor in range(0, len(cols), q):
            total = 0
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j != anchor:
                        w = (pair ^ cols[j]).bit_count()
                        if w <= 4:
                            total += (5 - w) ** 2
            burdens.append(total)
        return max(burdens), sum(burdens)

    def a_eight(cols):
        """Exact anchored four/four collision exposure, not a pL model.

        Called only after exclusion through six. Odd kernels have
        already been excluded by the verified parity-check combination.
        Distinct colliding four-supports are consequently disjoint.
        Counts weight-eight partition exposure, not a full spectrum.
        """
        width = len(cols)
        anchors = tuple(range(0, width, q))
        anchored = {}
        for anchor in anchors:
            allowed = [i for i in range(width)
                       if i != anchor and i not in anchors[:anchors.index(anchor)]]
            for bi, b in enumerate(allowed):
                ab = cols[anchor] ^ cols[b]
                for ci in range(bi + 1, len(allowed)):
                    c = allowed[ci]
                    abc = ab ^ cols[c]
                    for d in allowed[ci + 1:]:
                        value = abc ^ cols[d]
                        anchored[value] = anchored.get(value, 0) + 1

        exposure = -sum(anchored.values())
        get = anchored.get
        for a in range(width - 3):
            for b in range(a + 1, width - 2):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, width - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, width):
                        exposure += get(abc ^ cols[d], 0)
        assert exposure >= 0
        return exposure

    apool = []
    for serial, A in enumerate(book):
        _, coarse, rows, cols = analyze("A", A)
        bad6 = _q42_small_kernel(cols, True)
        key = (coarse[0], coarse[1], int(bad6),
               coarse[2], coarse[4], serial)
        apool.append((key, A, rows, cols, coarse))
    apool.sort(key=lambda item: item[0])

    aa = []
    for base, A, rows, cols, coarse in apool[:3]:
        bad6 = bool(base[2])
        exposure = 10**12 if bad6 else a_eight(cols)
        distance, low, peak = _q42_sample(rng, cols, q, 48)
        boundary = anchored_boundary(cols)
        aa.append({
            "grid": A, "cols": cols, "coarse": coarse,
            "bad6": int(bad6), "eight": exposure,
            "distance": distance, "low": low, "peak": peak,
            "boundary": boundary,
        })

    bpool, seen = [], set()

    def retain_B(B, state):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        _, coarse, rows, cols = state
        bpool.append((coarse, len(seen), clone(B), rows, cols))
        bpool.sort(key=lambda item: item[:2])
        del bpool[6:]

    positions = [(i, j) for i in range(2) for j in range(2, 6)]
    positions.append((1, 1))
    for restart in range(4):
        B = draw_B()
        current = analyze("B", B)
        retain_B(B, current)
        best_grid, best = clone(B), current
        for step in range(200):
            i, j = rng.choice(positions)
            old = B[i][j]
            B[i][j] = b_cell(i, j)
            proposed = analyze("B", B)
            delta = proposed[0] - current[0]
            temperature = 18.0 * (0.975 ** step) + 0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                current = proposed
                if current[0] < best[0]:
                    best_grid, best = clone(B), current
            else:
                B[i][j] = old
            if step in (99, 199):
                retain_B(B, current)
        retain_B(best_grid, best)

    bb = []
    for coarse, serial, B, rows, cols in bpool[:4]:
        bad4 = _q42_small_kernel(cols, False)
        sparse = [c for c in cols if c.bit_count() == 2]
        sparse_bad6 = _q42_small_kernel(sparse, True)
        distance, low, peak = _q42_sample(rng, cols, q, 36)
        boundary = anchored_boundary(cols)
        key = (coarse[0], int(bad4), int(sparse_bad6),
               int(distance < 10), low, peak,
               boundary, coarse[3:], coarse[2], serial)
        bb.append((key, {
            "grid": B, "cols": cols, "coarse": coarse,
            "bad4": int(bad4), "sparse_bad6": int(sparse_bad6),
            "distance": distance, "low": low, "peak": peak,
            "boundary": boundary,
        }))
    bb.sort(key=lambda item: item[0])
    bb = [item[1] for item in bb[:3]]

    options = []
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 275*n <= 1000*k < 325*n):
                continue
            ac, bc = a["coarse"], b["coarse"]
            # Match a coarse protection floor first. Then prioritize
            # A's collective ambiguity, because the parent was usually
            # Z-limited. Upper-bound distances are not circuit scores.
            key = (
                ac[0] + bc[0], ac[1] + b["bad4"],
                a["bad6"], b["sparse_bad6"],
                int(min(a["distance"], b["distance"]) < 10),
                a["eight"],
                max(a["low"] / 84, b["low"] / 252),
                max(a["peak"] / 84, b["peak"] / 252),
                max(a["boundary"][0], b["boundary"][0]),
                a["boundary"][1] + b["boundary"][1],
                ac[3] + bc[3], ac[4] + bc[4], k - 168,
            )
            options.append((key, A, B, hx, hz, n, k))
    options.sort(key=lambda item: item[0])

    def emit(item):
        key, A, B, hx, hz, n, k = item
        assert n == 588
        assert css_ok(hx, hz)
        assert {row.bit_count() for row in hx} == {10, 11}
        assert {row.bit_count() for row in hz} == {12}
        assert 6 * sum(row.bit_count() for row in hx + hz) == 28224
        assert all(len(cell) <= 5 and len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        actual_a, actual_b = _f21_columns(hx, hz, q)
        assert actual_a == component("A", A)[1]
        assert actual_b == component("B", B)[1]
        assert _q42_cycles(component("A", A)[0], actual_a)[0] == 0
        return {"idx": idx, "contract": (a_card, policy),
                "k": k, "screen": key, "cells": {"A": A, "B": B}}

    for item in options:
        _, _, _, hx, hz, n, _ = item
        if not (_q42_low4(hx, hz, n) or _q42_low4(hz, hx, n)):
            return emit(item)
    if options:
        return emit(options[0])

    # Preserve the indexed A panel and B contract in the legal fallback.
    for attempt in range(96):
        A, B = clone(rng.choice(book)), draw_B()
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k > 0 and 275*n <= 1000*k < 325*n:
            return emit(((10**12, attempt), A, B, hx, hz, n, k))
    raise ValueError("no compensated fourton filling retained cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected parity-fourton/long-holonomy cell unchanged."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
# REFUTED: Increasing d2s necessarily improves this bed's circuit score: zero-failure slot 16 had d2s=21,k=23,s=6.874, below zero-failure slot 28 with d2s=18,k=30,s=6.989; their sample ceilings limited both.
# REFUTED: Rank-one C3 quotient singleton/triple fillings reliably retain the earlier order-21 bed's zero-failure memory while raising k: current slot12 had 100/129750 Z failures and s=5.584, slot15 had 37/200000 X failures, and slot18 produced no ASC schedule.
# REFUTED: Prime-order C19 with connected full-rank components and ambiguity screening suffices for rare circuit failures: all five current-parent measured d2s=8 codes reached failure caps, scored 5.226..5.614, and collectively flipped all 171 logicals.
# REFUTED: The baseline order-42 parameters n=588,k=176,w=12,d2s=10 establish uniform circuit reliability: slot8 scored 5.772 with 100/68250 Z failures, while slot10 scored 6.443 with 59/200000 Z failures at the same reported parameters and 25704 CNOTs.
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
