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
CONCEPT = ("C37 semidirect C1 with action 1, order 37, shapes (2,3)x(2,4): cancellation-screened cofactor witnesses and independently optimized mixed-weight two-row components should improve cell-0.150 measured five-code fitness above 44.1.")
GROUP = ("L2 order repair within the unchanged metacyclic construction rule: (m,r,s)=(37,1,1). The new size factor sixteen makes order forty inadmissible; order thirty-seven is maximal and gives n=592. The cyclic-order-37 bed on these shapes measured fitness 44.1 with distance-eighteen leaders. Commutativity permits constructive cofactor words, but every syndrome, component rank and assembled rank is obtained from verified scaffold actions.")
SHAPES = ("Target cell 0.150 with (2,3)x(2,4): n=592, kb=74, guaranteed rate 1/8, maximum lift order 37, and component variable/check counts 111/74 and 148/74. Both components now have two check rows. Parent weight-five X minima occupied sector-one Acol/Bcol blocks, especially Acol1; its Z minima weighed six. The new second sector has 148 rather than 40 qubits. Exact assembled ranks enforce 0.125<=k/n<0.175, including any rank surplus.")
FILLING = ("Twenty cards cross five doubleton, cubic/singleton and quartet/zero support architectures with four intrinsic relative-gap models. Both CSS weight maxima are at most twelve and every protograph line remains active. Compile components exclusively from verified unit-cell lifts, construct ordinary and eligible common-binomial-divided cofactors, and reject their actual cancellations before bounded true-kernel searches. Independently repair and deepen component finalists, then classify embedded and mixed-sector words modulo exact stabilizer spans. Require complete logical-four exclusion and fresh uncoloured Tanner invariants. Selection uses found non-stabilizer witnesses, never a no-witness sentinel or a distance certificate inferred from a search miss. No archived parent bank is called.")
FAMILY_NAME = "r015_metacyclic_C37sdC1_23x24_mixedsupport_dividedcofactor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Same metacyclic rule; size-law repair to m=37, r=1, s=1.
    if not valid_q(q):
        raise ValueError("C37 semidirect C1 requires order 37")
    m, r, s = 37, 1, 1
    assert m*r == q and pow(s, r, m) == 1
    els = [(a, b) for b in range(r) for a in range(m)]
    action = [pow(s, b, m) for b in range(r)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + action[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-action[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return a + m * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _four_cycle_count(rows):
    """Four-cycles in one Tanner graph, using the verified bit rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _has_small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight 1..4.

    Enumerate supports of size at most two. Equal syndromes yield
    every commuting support of size at most four. Test differences
    against the opposite stabilizer row space, not merely the kernel.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    syndrome = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndrome[low.bit_length() - 1] ^= bit
            row ^= low

    seen = {0: 0}
    for i, s in enumerate(syndrome):
        v = 1 << i
        old = seen.get(s)
        if old is not None:
            if outside(v ^ old):
                return True
        else:
            seen[s] = v

    for i in range(n):
        vi = 1 << i
        si = syndrome[i]
        for j in range(i):
            s = si ^ syndrome[j]
            v = vi | (1 << j)
            old = seen.get(s)
            if old is not None:
                if outside(v ^ old):
                    return True
            else:
                seen[s] = v
    return False


@__import__("functools").lru_cache(maxsize=1)
def _shared_cokernel_bank(G):
    """Pure, cached optimization of fresh rank-aware design cards."""
    from collections import Counter

    q = G.q
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(3)]]
    dummy_b = [[[G.e] for _ in range(4)]]

    def component_rows(cells, side):
        if side == "A":
            hx, hz, n = lifted_matrices(G, cells, dummy_b)
            rows = []
            for row in hx[:q]:
                rows.append(sum(
                    ((row >> (4*j*q)) & mask) << (j*q)
                    for j in range(3)))
            return rows, 3*q
        hx, hz, n = lifted_matrices(G, dummy_a, cells)
        bm = (1 << (4*q)) - 1
        return [row & bm for row in hz[:q]], 4*q

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def dependency_space(rows):
        """Canonical exact space of dependencies between verified rows."""
        pivots, dependencies = {}, []
        for r, original in enumerate(rows):
            row, word = original, 1 << r
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (row, word)
                    break
                row ^= old[0]
                word ^= old[1]
            if not row:
                dependencies.append(word)

        basis = {}
        for word in dependencies:
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def fingerprint(hx, hz, n):
        """Permutation-invariant combined, uncoloured incidence signature.

        Different signatures imply different incidence graphs. Equal
        signatures are conservatively rejected, not asserted isomorphic.
        """
        rows = list(hx) + list(hz)
        adj = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adj[i][j] = overlap
                    adj[j][i] = overlap
        profiles = []
        for r, neighbors in enumerate(adj):
            items = list(neighbors.items())
            triangles = 0
            for a, (s, ws) in enumerate(items):
                for t, wt in items[:a]:
                    triangles += 2*ws*wt*adj[s].get(t, 0)
            profiles.append((
                rows[r].bit_count(),
                tuple(sorted(Counter(neighbors.values()).items())),
                sum(w*w for w in neighbors.values()),
                triangles))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(degrees).items())))

    def canonical_space(words):
        """Canonical binary row space; no independent lift convention."""
        basis = {}
        for word in words:
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def intersection(first, second):
        """Exact intersection of two verified row-dependency spaces."""
        pivots = {
            word.bit_length() - 1: (word, 0) for word in first
        }
        common = []
        for original in second:
            syndrome, word = original, original
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                common.append(word)
        return canonical_space(common)

    # Every cubic cell can be right-normalized to contain the identity.
    # Right multiplication changes only its left-lift column permutation.
    # Enumerating all normalized triples therefore removes gauge draws
    # without omitting a single-block column-incidence geometry.
    classes = {}
    for a in range(1, q):
        for b in range(a + 1, q):
            cell = (G.e, G.elements[a], G.elements[b])
            A = [[list(cell), [G.e], [G.e]]]
            hx, hz, n = lifted_matrices(G, A, dummy_b)
            rows = [row & mask for row in hx[:q]]
            deficit = q - gf2_rank(rows)
            # Do not discard higher singularity strata before testing
            # their actual kernels and compatibility with other blocks.
            if deficit < 4:
                continue
            if _has_small_logical(rows, [], q):
                continue
            space = dependency_space(rows)
            incidence = tuple(sorted(columns(rows, q)))
            classes.setdefault(space, {}).setdefault(incidence, cell)

    spaces = tuple(classes)
    common_spaces = set(spaces)
    for i, first in enumerate(spaces):
        for second in spaces[:i]:
            # Rank arithmetic is only a cheap intersection-size filter.
            dimension = (len(first) + len(second)
                         - gf2_rank(first + second))
            if dimension < 4:
                continue
            common = intersection(first, second)
            assert len(common) == dimension
            common_spaces.add(common)

    # A shared dependency space need not equal any individual block's
    # entire dependency space. Larger singular blocks may contain it.
    packed = []
    for common in sorted(common_spaces,
                         key=lambda s: (-len(s), s)):
        cells = {}
        for space, bucket in classes.items():
            if len(space) < len(common):
                continue
            if gf2_rank(space + common) != len(space):
                continue
            for incidence, cell in bucket.items():
                cells.setdefault(incidence, cell)
        if len(cells) >= 3:
            packed.append((common, tuple(cells.values())))

    library = tuple(sorted(
        packed, key=lambda item: (-len(item[0]),
                                  -len(item[1]), item[0])))

    # Four odd-voltage geometries, crossed with five intrinsic cubic
    # quotient occupancies. No card is made by translating another code.
    odd_pairs = ((1, 3), (1, 5), (3, 7), (5, 7))
    b_models = ((0, 2, 4), (0, 3, 6), (0, 1, 6),
                (0, 2, 7), (0, 3, 5))
    target_deficits = (16, 12, 8, 4)
    target_spreads = (6, 7, 8, 9)
    fingerprints, bank = set(), []

    def generators(rng, policy, mode, fallback):
        b0, b1 = odd_pairs[policy]
        u = rng.randrange(5)
        # Distinct complement parameters, not just different labels.
        denominator = (1 - pow(2, b0, 5)) % 5
        parameter = u * pow(denominator, -1, 5) % 5
        forbidden = (1 - pow(2, b1, 5)) * parameter % 5
        v = rng.choice([a for a in range(5) if a != forbidden])
        # (a,4), a!=0, has order ten; (a,0) has order five.
        # The broader fallback changes the voltage geometry, not counts.
        even_b = 4 if mode % 2 else 0
        if fallback:
            even_b = 4 - even_b
        return ((u, b0), (v, b1),
                (rng.randrange(1, 5), even_b))

    def triple(rng, model):
        used, cell = set(), []
        for b in model:
            choices = [(a, b) for a in range(5) if (a, b) not in used]
            g = rng.choice(choices)
            used.add(g)
            cell.append(g)
        base = G.pick(rng)
        return [G.mul(base, g) for g in cell]

    def draw_b(rng, policy, mode, fallback):
        gs = generators(rng, policy, mode, fallback)
        return [[[G.e, g] for g in gs]
                + [triple(rng, b_models[mode])]]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    for slot in range(N_SLOTS):
        card = slot % 20
        policy, mode = divmod(card, 5)
        rng = random.Random(40013691 + 100003*slot)
        winner = None

        # Prescribed high-deficit tiers are attempted first, but smaller
        # clean containment groups remain available in the broad search.
        eligible = [
            item for item in library
            if len(item[0]) >= target_deficits[policy]]
        if not eligible and library:
            largest = len(library[0][0])
            eligible = [item for item in library
                        if len(item[0]) == largest]

        for fallback in (False, True):
            choices = library if fallback else eligible
            if not choices:
                continue
            sa = [[3, 3, 3]]
            sb = [[2, 2, 2, 3]]
            assert _grid_weights(sa, sb) <= 12
            assert _no_dead_lines(sa, sb)

            apool, aseen = [], set()
            quota = 256 if fallback else 192
            for attempt in range(quota):
                space, cells = rng.choice(choices)
                A = [[list(cell) for cell in rng.sample(cells, 3)]]
                # A-column permutations cannot spend proposal diversity.
                sig = tuple(sorted(
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in A[0]))
                if sig in aseen:
                    continue
                aseen.add(sig)
                ar, width = component_rows(A, "A")
                if _has_small_logical(ar, [], width):
                    continue
                deficit = q - gf2_rank(ar)
                spread = sum(len({g[1] for g in cell}) for cell in A[0])
                key = (deficit,
                       -abs(spread - target_spreads[policy]),
                       -_four_cycle_count(ar))
                apool.append((key, A))
                apool.sort(key=lambda item: item[0], reverse=True)
                del apool[6:]
                if len(apool) >= 6 and attempt >= 96:
                    break
            if not apool:
                continue

            bpool, bseen = [], set()
            for attempt in range(192):
                B = draw_b(rng, policy, mode, fallback)
                sig = signature(dummy_a, B)
                if sig in bseen:
                    continue
                bseen.add(sig)
                br, width = component_rows(B, "B")
                if _has_small_logical(br, [], width):
                    continue
                key = (_four_cycle_count(br), -gf2_rank(br))
                bpool.append((key, B))
                bpool.sort(key=lambda item: item[0])
                del bpool[6:]
                if len(bpool) >= 6 and attempt >= 48:
                    break
            if not bpool:
                continue

            proposals = []
            for ak, A in apool:
                for bk, B in bpool:
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (475*n <= 1000*k < 525*n):
                        continue
                    cycles = -ak[2] + bk[0]
                    # Exact logical-four screening establishes only the
                    # exclusion through four. The evaluator must still
                    # measure the distance-five-or-higher hypothesis.
                    key = (k, -cycles, ak[1])
                    proposals.append((key, A, B, hx, hz, n))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, A, B, hx, hz, n in proposals[:12]:
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in fingerprints:
                    continue
                assert css_ok(hx, hz), "verified lift failed CSS commutation"
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert _grid_weights(
                    [[len(cell) for cell in row] for row in A],
                    [[len(cell) for cell in row] for row in B]) <= 12
                winner = (A, B, k, fp, fallback)
                break
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B, k, fp, fallback = winner
        fingerprints.add(fp)
        bank.append({
            "cells": {"A": A, "B": B}, "k": k,
            "card": card, "fallback": fallback,
            "screen_no_le4": True})

    return tuple(bank)


# The historical half-rate bank above is inactive. Only the following
# shape-aware optimizer supplies cells for this program.
_R15_PROFILES = (
    (((2,2,2), (2,2,2)), ((3,3,1,1), (1,1,3,3))),
    (((2,2,2), (2,2,2)), ((3,2,2,1), (1,2,2,3))),
    (((2,2,2), (2,2,2)), ((4,2,1,1), (0,2,3,3))),
    (((2,2,2), (1,2,3)), ((3,2,1,1), (1,1,2,3))),
    (((3,2,1), (1,2,3)), ((3,3,1,1), (1,1,3,3))),
)


def _r15_columns(rows, width):
    out = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            out[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return tuple(out)


def _r15_basis(words):
    out = {}
    for word in words:
        while word:
            p = word.bit_length()-1
            old = out.get(p)
            if old is None:
                out[p] = word
                break
            word ^= old
    return out


def _r15_outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _r15_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _r15_components(hx, hz, q):
    """Extract the actual component conventions from a complete LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = [
        sum(((hx[(i*cb)*q+r] >> (j*cb*q)) & mask) << (j*q)
            for j in range(ca))
        for i in range(ra) for r in range(q)]
    br = [
        hz[a*q+r] & ((1 << (cb*q))-1)
        for a in range(rb) for r in range(q)]
    return ar, br


def _r15_draw(local, G, counts, mode):
    """Different relative-gap architectures, followed by fresh offsets."""
    q, width = G.q, len(counts[0])
    if mode == 0:
        top = [local.randrange(1, q) for _ in range(width)]
        bottom = [local.randrange(1, q) for _ in range(width)]
    elif mode == 1:
        top = local.sample(range(1, q), width)
        bottom = list(top)
        local.shuffle(bottom)
    else:
        templates = (
            ((1,2,4,8), (3,5,9,13)),
            ((1,3,7,12), (2,6,11,17)))
        first, second = templates[mode-2]
        unit = local.randrange(1, q)
        top = [unit*d % q for d in first[:width]]
        bottom = [unit*d % q for d in second[:width]]

    grid = [[[] for _ in range(width)] for _ in range(2)]
    for j in range(width):
        for i in range(2):
            count = counts[i][j]
            if not count:
                continue
            base = 0 if i == 0 else local.randrange(q)
            gap = (top if i == 0 else bottom)[j]
            residues = [base]
            if count >= 2:
                residues.append((base+gap) % q)
            while len(residues) < count:
                if mode == 1:
                    candidate = (base+len(residues)*gap) % q
                else:
                    candidate = local.randrange(q)
                if candidate not in residues:
                    residues.append(candidate)
            grid[i][j] = [(a, 0) for a in residues]
    return grid


def _r15_cofactors(G, grid, cols, inverse):
    """Construct words, not another lift; verify every physical syndrome."""
    from itertools import combinations
    q, mask = G.q, (1 << G.q)-1
    entries = [
        [sum(1 << G.idx(G.inv(g) if inverse else g) for g in cell)
         for cell in row] for row in grid]

    def product(a, b):
        answer = 0
        while a:
            bit = a & -a
            t = bit.bit_length()-1
            answer ^= ((b << t) | (b >> (q-t))) & mask
            a ^= bit
        return answer

    def divide(polynomial):
        quotient = 0
        while polynomial:
            p = polynomial.bit_length()-1
            if p == 0:
                raise ValueError("nonzero remainder in even-cell division")
            quotient ^= 1 << (p-1)
            polynomial ^= 3 << (p-1)
        return quotient

    matrices = [entries]
    if all(p.bit_count() % 2 == 0 for row in entries for p in row):
        matrices.append([[divide(p) for p in row] for row in entries])

    found = set()
    width = len(grid[0])
    for matrix in matrices:
        for triple in combinations(range(width), 3):
            word = 0
            for omitted in triple:
                s, t = [j for j in triple if j != omitted]
                minor = (product(matrix[0][s], matrix[1][t])
                         ^ product(matrix[0][t], matrix[1][s]))
                word ^= minor << (omitted*q)
            if word and _r15_syndrome(cols, word) == 0:
                found.add(word)

    # Exact constant-coordinate restriction, including mixed augmentations.
    indicators = [mask << (j*q) for j in range(width)]
    syndromes = [_r15_syndrome(cols, word) for word in indicators]
    for support in range(1, 1 << width):
        word, syndrome = 0, 0
        for j in range(width):
            if (support >> j) & 1:
                word ^= indicators[j]
                syndrome ^= syndromes[j]
        if not syndrome:
            found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w)))


def _r15_probe(cols, local, trials, seeds=(), stabilizers=None,
               orbit_q=None):
    """Retain true witnesses; quantum readings exclude the stabilizer span."""
    span = None if stabilizers is None else _r15_basis(stabilizers)
    found = set()
    memory = sorted(
        {w for w in seeds if w and _r15_syndrome(cols, w) == 0},
        key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))

    def observe(word):
        if word and (span is None or _r15_outside(word, span)):
            found.add(word)

    for word in memory:
        observe(word)

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

        for sweep in range(2):
            reducers = sorted(set(dependencies+memory),
                              key=int.bit_count)[:28]
            changed = False
            for i, word in enumerate(dependencies):
                weight = word.bit_count()
                for reducer in reducers:
                    candidate = word ^ reducer
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                        changed = True
                dependencies[i] = word
                observe(word)
            if not changed:
                break

        discoveries = set(dependencies+memory)
        short = sorted(discoveries, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        if stabilizers is not None:
            # Logical directions have separate storage; short check rows
            # must not crowd every nontrivial class out of the search.
            logicals = sorted(found, key=int.bit_count)[:6]
            for start in logicals:
                word = start
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda x: x[0])[:12]
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    for reducer in short:
                        candidate = word ^ reducer
                        cw = candidate.bit_count()
                        if (0 < cw < weight
                                and _r15_outside(candidate, span)):
                            chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)

        found = set(sorted(found, key=int.bit_count)[:32])
        memory = sorted(discoveries, key=int.bit_count)[:32]

    if orbit_q is not None and found:
        # Translate witnessed words only. Check the actual syndrome,
        # rather than assume which regular-action convention commutes.
        q, mask = orbit_q, (1 << orbit_q)-1
        width = len(cols)//q
        parents = sorted(found, key=int.bit_count)[:5]
        images = set()
        for word in parents:
            for t in range(q):
                moved = sum(
                    ((((word >> (j*q)) & mask) << t
                      | (((word >> (j*q)) & mask) >> (q-t))) & mask)
                    << (j*q) for j in range(width))
                if moved and _r15_syndrome(cols, moved) == 0:
                    images.add(moved)
        for word in parents:
            for other in images:
                candidate = word ^ other
                if candidate:
                    observe(candidate)

    answer = tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:32])
    # No-witness sentinels never enter an optimization score.
    return answer


def _r15_signature(hx, hz, n):
    """Combined uncoloured Tanner invariant; inequality proves distinction."""
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    neighbors = [dict() for _ in rows]
    adjacency = [0] * len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                neighbors[i][j] = neighbors[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = []
    for i, local in enumerate(neighbors):
        histogram = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in local.items())
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    cols = _r15_columns(rows, n)
    qubit_profiles = []
    for col in cols:
        profile, remaining = [], col
        while remaining:
            bit = remaining & -remaining
            profile.append(weights[bit.bit_length()-1])
            remaining ^= bit
        qubit_profiles.append(tuple(sorted(profile)))
    return (n, tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(qubit_profiles).items())))


@__import__("functools").lru_cache(maxsize=1)
def _r15_bank(G):
    """Pure bank of fresh, independently optimized two-row design cards."""
    assert G.q == 37 and A_SHAPE == (2,3) and B_SHAPE == (2,4)
    q, mask = G.q, (1 << G.q)-1
    za = [[[] for _ in range(3)] for _ in range(2)]
    zb = [[[] for _ in range(4)] for _ in range(2)]

    # Compile only verified unit-cell actions. The remaining compilation
    # consists of their F2-linear superposition and block placement.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(_r15_columns(
                [row & mask for row in source[:q]], q))
        atlas[side] = tuple(blocks)

    def component(side, grid):
        width = len(grid[0])*q
        cols = [0] * width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for h, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+h] ^= value << (i*q)
        rows = [0] * (2*q)
        for c, original in enumerate(cols):
            value = original
            while value:
                bit = value & -value
                rows[bit.bit_length()-1] |= 1 << c
                value ^= bit
        return rows, tuple(cols)

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        profile, mode = divmod(slot % 20, 4)
        sa, sb = _R15_PROFILES[profile]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(37150291 + 104729*slot)

        def component_key(item):
            d = item["words"][0].bit_count()
            load = sum(w.bit_count() <= d+1 for w in item["words"])
            return (min(d, 28), -load, -item["cycles"])

        def pool(side, counts, batch):
            raw, seen = [], set()
            aug = [sum((count & 1) << j for j, count in enumerate(row))
                   for row in counts]
            # This rank gate is a constituent measurement, not a homology
            # formula: retain full nontrivial cyclic-frequency rank.
            wanted_rank = 2*q-2+gf2_rank(aug)

            def admit(grid):
                labels = tuple(tuple(sorted(G.idx(g) for g in cell))
                               for row in grid for cell in row)
                if labels in seen:
                    return
                seen.add(labels)
                rows, cols = component(side, grid)
                if gf2_rank(rows) != wanted_rank:
                    return
                seeds = _r15_cofactors(G, grid, cols, side == "B")
                if seeds and seeds[0].bit_count() < 10:
                    return
                words = _r15_probe(cols, local, 2, seeds)
                if not words or words[0].bit_count() <= 4:
                    return
                item = dict(grid=grid, rows=rows, cols=cols,
                            words=words, cycles=_four_cycle_count(rows))
                raw.append(item)
                raw.sort(key=component_key, reverse=True)
                del raw[8:]

            for attempt in range(80+32*batch):
                admit(_r15_draw(local, G, counts, mode))
            for step in range(28):
                if not raw:
                    admit(_r15_draw(local, G, counts, mode))
                    continue
                parent = local.choice(raw[:min(4, len(raw))])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                fresh = _r15_draw(local, G, counts, mode)
                width = len(counts[0])
                pressure = [
                    1+sum(((word >> (j*q)) & mask).bit_count()
                          for word in parent["words"][:8])
                    for j in range(width)]
                j = local.choices(range(width), weights=pressure, k=1)[0]
                i = local.randrange(2)
                grid[i][j] = fresh[i][j]
                if step % 5 == 4:
                    grid[1-i][j] = fresh[1-i][j]
                admit(grid)

            for item in raw:
                item["words"] = _r15_probe(
                    item["cols"], local, 16, item["words"], orbit_q=q)
            raw = [item for item in raw if item["words"]]
            raw.sort(key=component_key, reverse=True)
            leaders = raw[:4]
            for item in leaders:
                item["words"] = _r15_probe(
                    item["cols"], local, 64, item["words"], orbit_q=q)
            leaders.sort(key=component_key, reverse=True)
            return leaders

        def merit(item):
            d = min(item["a"]["words"][0].bit_count(),
                    item["b"]["words"][0].bit_count(),
                    item.get("dx", 593), item.get("dz", 593), 28)
            return (item["k"]*d*d/592, d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        chosen = None
        for batch in range(3):
            ap = pool("A", sa, batch)
            bp = pool("B", sb, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 592 or not (125*n <= 1000*k < 175*n):
                        continue
                    ar, br = _r15_components(hx, hz, q)
                    assert _r15_columns(ar, 3*q) == a["cols"]
                    assert _r15_columns(br, 4*q) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            proposals.sort(key=merit, reverse=True)

            finalists = []
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fp = _r15_signature(hx, hz, n)
                if fp in emitted:
                    continue
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue

                # These sector-one embeddings are checked against the
                # actual complete matrices before entering the search.
                aw = [
                    sum(((word >> (j*q)) & mask)
                        << ((4*j+beta)*q) for j in range(3))
                    for word in item["a"]["words"][:12]
                    for beta in range(4)]
                bw = [
                    word << (j*4*q)
                    for word in item["b"]["words"][:12]
                    for j in range(3)]
                xc = _r15_columns(hz, n)
                zc = _r15_columns(hx, n)
                assert all(_r15_syndrome(zc, w) == 0 for w in aw)
                assert all(_r15_syndrome(xc, w) == 0 for w in bw)
                wx = _r15_probe(xc, local, 3, bw, stabilizers=hx)
                wz = _r15_probe(zc, local, 3, aw, stabilizers=hz)
                if not wx or not wz:
                    continue
                item.update(xc=xc, zc=zc, wx=wx, wz=wz,
                            dx=wx[0].bit_count(), dz=wz[0].bit_count(),
                            signature=fp)
                finalists.append(item)
                if len(finalists) >= 3:
                    break

            # Equal fresh deeper budgets; earlier valid witnesses remain
            # in memory and cannot be erased by a favorable search miss.
            for item in finalists:
                wx = _r15_probe(
                    item["xc"], local, 12, item["wx"],
                    stabilizers=item["hx"])
                wz = _r15_probe(
                    item["zc"], local, 12, item["wz"],
                    stabilizers=item["hz"])
                if wx:
                    item["dx"] = min(item["dx"], wx[0].bit_count())
                if wz:
                    item["dz"] = min(item["dz"], wz[0].bit_count())
            if not finalists:
                continue
            winner = max(finalists, key=merit)
            A, B = winner["A"], winner["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert css_ok(hx, hz)
            assert winner["k"] == lifted_k(G, A, B)
            assert 125*n <= 1000*winner["k"] < 175*n
            actual_a = [[len(c) for c in row] for row in A]
            actual_b = [[len(c) for c in row] for row in B]
            assert actual_a == [list(row) for row in sa]
            assert actual_b == [list(row) for row in sb]
            assert _grid_weights(actual_a, actual_b) <= 12
            assert _no_dead_lines(actual_a, actual_b)
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert all(len(c) == len(set(c)) <= 5
                       for grid in (A, B) for row in grid for c in row)
            emitted.add(winner["signature"])
            chosen = dict(
                cells={
                    side: tuple(tuple(tuple(c) for c in row) for row in grid)
                    for side, grid in (("A", A), ("B", B))},
                k=winner["k"], card=(profile, mode),
                selection=merit(winner), screen_no_le4=True)
            break
        # Failed cards never call the archived half-rate bank.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return every cell of this fresh target-rate two-row design."""
    selected = _r15_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rate-0.150 two-row filling passed screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel/diversity cards guarantee distinct incidence geometries per slot; measured parent slots 4,5,12,13 were iso_dup_of_0 and slot 16 was iso_dup_of_1.
# REFUTED: Dependency-containment packing produces five k>=264 leaders on the order-40 (1,3)x(1,4) bed; all five measured current leaders instead had k=252 and d30s=5, leaving fitness at 12.1150.
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
