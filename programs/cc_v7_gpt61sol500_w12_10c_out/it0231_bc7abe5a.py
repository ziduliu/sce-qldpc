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
CONCEPT = ("C9 semidirect C4 with inversion action, order 36, shapes (2,3)x(2,4): degree-distinguished supports, retained subgroup/component witnesses and stabilizer-aware full rescreening should exceed the 0.150 frontier fitness 40.8.")
GROUP = ("Preserve the split-metacyclic construction rule, changing only m,r,s from 13,3,3 to 9,4,8. The new shape would give n=624 at order 39, so reduce to order 36. C9:C4 with inversion is isomorphic to Dic9: u=(1,2), v=(0,1) satisfy u^18=1, v^2=u^9 and v*u*v^-1=u^-1. The shown Dic9 two-check program measured 38.9336; RUN STATE's comparable order-36 bed reached 40.8.")
SHAPES = ("Target cell 0.150 with A=(2,3), B=(2,4): n=576, kb=72, guaranteed rate 1/8, maximum lift order 37. Constituent variables/checks are 108/72 and 144/72. Both retain two check rows. Parent measured weight-22 minima were entirely sector 1, emphasizing Acol1/2 and Bcol1/2, with no Arow/Brow minima. Verified full ranks enforce 0.125<=k/n<0.175.")
FILLING = ("Enumerate canonical support pairs with distinct combined check/qubit degree profiles and both row caps <=12. Explore order-18, order-4 and order-9 doubletons plus four C4 occupancy rules for larger cells; exclude central-involution pairs and complete C3 cosets. Screen subgroup-fiber words and retain actual kernel supports across randomized rescreens. Repair witness-heavy columns, independently deepen finalists, reject logical weights <=4 exactly, and search stabilizer-tested full logical witnesses in both sectors. Component readings are selection heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r015_C9sdC4_23x24_degree_norm_retained_full_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # Same split-metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    m, r, s = 9, 4, 8
    if q != m*r:
        raise ValueError("C9:C4 requires order 36")
    assert pow(s, r, m) == 1
    scales = tuple(pow(s, b, m) for b in range(r))
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return m*b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + scales[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-scales[bi]*a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _cycle_counts(rows):
    """Four/six cycles of a binary component Tanner graph."""
    size = len(rows)
    overlap = [[0]*size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i):
            t = (rows[i] & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = t
            four += t*(t-1)//2
    six = 0
    for i in range(size):
        for j in range(i):
            u = overlap[i][j]
            if not u:
                continue
            common = rows[i] & rows[j]
            for k in range(j):
                v, w = overlap[j][k], overlap[i][k]
                if v and w:
                    t = (common & rows[k]).bit_count()
                    six += u*v*w - t*(u+v+w-2)
    return four, six


def _component_objective(G, A, B, design):
    # Extract components only from the scaffold's verified matrices.
    hx, hz, n = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q)-1
    ar = []
    for i in range(2):
        for r in range(q):
            row = hx[i*3*q+r]
            ar.append(sum(((row >> (j*3*q)) & mask) << (j*q)
                          for j in range(3)))
    br = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
    a4, a6 = _cycle_counts(ar)
    b4, b6 = _cycle_counts(br)
    # Later slots also explore different A/B priorities and cycle tradeoffs.
    priority = design % 3
    wa, wb = ((1, 1), (2, 1), (1, 2))[priority]
    six_weight = (1, 2, 4)[(design//3) % 3]
    score = (1000*max(a4, b4) + 200*(wa*a4+wb*b4)
             + six_weight*(wa*a6+wb*b6))
    return score, hx, hz, n


def _row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]
    return basis


def _in_span(v, basis):
    while v:
        p = v.bit_length()-1
        if p not in basis:
            return False
        v ^= basis[p]
    return True


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1, 2, 3 or 4."""
    basis = _row_basis(stabilizers)
    syndromes = [0]*n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndromes[low.bit_length()-1] |= bit
            row ^= low
    singles = {}
    for i, s in enumerate(syndromes):
        v = 1 << i
        if s == 0 and not _in_span(v, basis):
            return True
        if s in singles:
            if not _in_span(v ^ singles[s], basis):
                return True
        else:
            singles[s] = v
    pairs = {}
    for i in range(n):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            s = si ^ syndromes[j]
            v = vi | (1 << j)
            if s == 0 and not _in_span(v, basis):
                return True
            if s in singles and not _in_span(v ^ singles[s], basis):
                return True
            if s in pairs:
                if not _in_span(v ^ pairs[s], basis):
                    return True
            else:
                pairs[s] = v
    return False


def _retired_23x23_ctx(rng, G, idx):
    """Unused parent policy; the shape-aware NEW_CTX below supersedes it."""
    from itertools import product, permutations, combinations

    # Canonical supports eliminate row/column permutation orbit copies.
    # Degree fingerprints distinguish the complete CSS Tanner graphs,
    # rather than just the component protographs.
    perms = tuple(permutations(range(3)))
    profiles = []
    for t in product(range(1, 5), repeat=6):
        rows = (sum(t[:3]), sum(t[3:]))
        cols = tuple(t[j] + t[j+3] for j in range(3))
        if min(rows) < 5 or max(rows) > 8:
            continue
        if min(cols) < 3 or max(cols) > 6:
            continue
        canonical = min(
            tuple(t[3*r+c] for r in order for c in perm)
            for order in ((0, 1), (1, 0)) for perm in perms)
        if t == canonical:
            profiles.append((t, rows, cols))

    pairs = []
    for a, (ta, ar, ac) in enumerate(profiles):
        for tb, br, bc in profiles[a:]:
            if max(ar) + max(bc) > W_MAX:
                continue
            if max(br) + max(ac) > W_MAX:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc] +
                [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc] +
                [x+y for x in ar for y in br]))
            priority = (min(ac+bc), min(ar+br), sum(ta)+sum(tb),
                        -max(ac+bc), -max(ar+br))
            pairs.append((priority, (checks, qubits), ta, tb))
    pairs.sort(reverse=True)
    fingerprints = set()
    designs = []
    for _, fingerprint, ta, tb in pairs:
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            designs.append((ta, tb))
    if len(designs) < 20:
        raise ValueError("fewer than twenty degree-distinguished designs")
    ta, tb = designs[idx % len(designs)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def contains_three_coset(cell):
        # Three points g, g*d, g*d^2 with d of order three.
        # This uses the actual group operations, not coordinate guesses.
        for g, h, u in combinations(cell, 3):
            d = G.mul(G.inv(g), h)
            d2 = G.mul(d, d)
            if (G.mul(d2, d) == G.e and
                    G.mul(g, d2) == u):
                return True
        return False

    def draw_grid(supports):
        grid = []
        for row in supports:
            out = []
            for weight in row:
                while True:
                    cell = rng.sample(G.elements, weight)
                    if not contains_three_coset(cell):
                        out.append(cell)
                        break
            grid.append(out)
        return grid

    def components(hx, hz):
        # Extract both actions from the VERIFIED lift.
        q = G.q
        mask = (1 << q)-1
        ar = []
        for i in range(2):
            for r in range(q):
                source = hx[i*3*q+r]
                ar.append(sum(
                    ((source >> (j*3*q)) & mask) << (j*q)
                    for j in range(3)))
        br = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
        return ar, br

    def columns(rows):
        cols = [0] * (3*G.q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return cols

    def probe(cols, trials):
        """Actual classical kernel witnesses, not a distance lower bound."""
        width = len(cols)
        best = width + 1
        for _ in range(trials):
            order = list(range(width))
            rng.shuffle(order)
            pivots = {}
            deps = []
            for c in order:
                syndrome, support = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    if p not in pivots:
                        pivots[p] = (syndrome, support)
                        break
                    old_syndrome, old_support = pivots[p]
                    syndrome ^= old_syndrome
                    support ^= old_support
                if not syndrome:
                    deps.append(support)
                    best = min(best, support.bit_count())

            # Invertible elementary basis operations preserve independent,
            # nonzero kernel vectors. Search pair sums as well.
            for _ in range(2):
                changed = False
                rng.shuffle(deps)
                for a in range(len(deps)):
                    v = deps[a]
                    weight = v.bit_count()
                    for b in range(len(deps)):
                        if a == b:
                            continue
                        z = v ^ deps[b]
                        zw = z.bit_count()
                        best = min(best, zw)
                        if 0 < zw < weight:
                            v, weight = z, zw
                            changed = True
                    deps[a] = v
                    best = min(best, weight)
                if not changed:
                    break
        return best

    pool = []
    exact_seen = set()
    for attempt in range(160):
        A, B = draw_grid(sa), draw_grid(sb)
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in exact_seen:
            continue
        exact_seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 8*k < n):
            continue
        ar, br = components(hx, hz)
        a4, a6 = _cycle_counts(ar)
        b4, b6 = _cycle_counts(br)
        cycle_key = (max(a4, b4), a4+b4, max(a6, b6), a6+b6)
        pool.append((cycle_key, attempt, k, A, B,
                     columns(ar), columns(br)))
        if len(pool) >= 48:
            break
    if not pool:
        raise ValueError("no positive-k labeling in target rate cell")

    # Do not assume that minimizing four-cycles also minimizes distance
    # witnesses: retain independent four- and six-cycle shortlists.
    shortlist = {}
    for candidate in sorted(pool, key=lambda c: (c[0], c[1]))[:8]:
        shortlist[candidate[1]] = candidate
    for candidate in sorted(
            pool, key=lambda c: (c[0][2:], c[0][:2], c[1]))[:8]:
        shortlist[candidate[1]] = candidate

    ranked = []
    for candidate in shortlist.values():
        cycles, attempt, k, A, B, ac, bc = candidate
        da, db = probe(ac, 24), probe(bc, 24)
        d = min(da, db)
        score = (k*d*d, d, da+db,
                 -cycles[0], -cycles[1], -cycles[3])
        ranked.append((score, attempt, da, db, candidate))
    ranked.sort(key=lambda item: item[:2], reverse=True)

    # Larger-budget rereadings prevent a lucky short probe from winning.
    refined = []
    for _, attempt, da, db, candidate in ranked[:6]:
        cycles, _, k, A, B, ac, bc = candidate
        da = min(da, probe(ac, 120))
        db = min(db, probe(bc, 120))
        d = min(da, db)
        score = (k*d*d, d, da+db,
                 -cycles[0], -cycles[1], -cycles[3])
        refined.append((score, attempt, A, B))
    refined.sort(key=lambda item: item[:2], reverse=True)

    for _, _, A, B in refined:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        assert k > 0 and 8*k < n
        if _small_logical(hz, hx, n) or _small_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        return {"A": A, "B": B}
    raise ValueError("refined designs all contain weight<=4 logicals")


@__import__("functools").lru_cache(maxsize=1)
def _lp_support_catalog():
    """Different degree profiles certify different combined Tanner graphs."""
    from itertools import combinations_with_replacement

    choices = tuple((a, b) for a in range(5) for b in range(5)
                    if 3 <= a+b <= 5)

    def profiles(width, low, high):
        out = []
        for columns in combinations_with_replacement(choices, width):
            mirror = tuple(sorted((b, a) for a, b in columns))
            if columns > mirror:
                continue
            rows = (sum(a for a, b in columns),
                    sum(b for a, b in columns))
            if min(rows) < low or max(rows) > high:
                continue
            totals = tuple(a+b for a, b in columns)
            counts = (tuple(a for a, b in columns)
                      + tuple(b for a, b in columns))
            out.append((counts, rows, totals))
        return out

    records = {}
    for ta, ar, ac in profiles(3, 5, 8):
        for tb, br, bc in profiles(4, 6, 9):
            if max(ar)+max(bc) > W_MAX:
                continue
            if max(br)+max(ac) > W_MAX:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc]
                + [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc]
                + [x+y for x in ar for y in br]))
            fingerprint = (checks, qubits)
            priority = (
                min(bc), min(ac), min(ar+br), sum(ta)+sum(tb),
                -abs(ar[0]-ar[1])-abs(br[0]-br[1]),
                -sum(w == 0 for w in ta+tb))
            record = (priority, ta, tb)
            old = records.get(fingerprint)
            if old is None or record > old:
                records[fingerprint] = record
    catalog = tuple((ta, tb) for _, ta, tb in
                    sorted(records.values(), reverse=True))
    if len(catalog) < N_SLOTS:
        raise ValueError("insufficient degree-distinguished support tasks")
    return catalog


def _lp_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _lp_components(hx, hz, q):
    """Extract both components from the scaffold's VERIFIED construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i*cb)*q+t]
            ar.append(sum(
                ((source >> (j*cb*q)) & mask) << (j*q)
                for j in range(ca)))
    br = [row & ((1 << (cb*q))-1) for row in hz[:rb*q]]
    return ar, br, _lp_columns(ar, ca*q), _lp_columns(br, cb*q)


def _lp_four(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap*(overlap-1)//2
    return total


def _lp_fibers(G):
    """Group operations construct test supports, never a replacement lift."""
    subgroups = {
        2: ((0, 0), (0, 2)),
        3: tuple((a, 0) for a in (0, 3, 6)),
        6: tuple((a, b) for b in (0, 2) for a in (0, 3, 6)),
        9: tuple((a, 0) for a in range(9)),
        18: tuple((a, b) for b in (0, 2) for a in range(9)),
    }
    result = {}
    for size, subgroup in subgroups.items():
        unseen, partition = set(range(G.q)), []
        while unseen:
            g = G.elements[min(unseen)]
            coset = {G.idx(G.mul(g, h)) for h in subgroup}
            assert len(coset) == size
            unseen.difference_update(coset)
            partition.append(sum(1 << t for t in coset))
        result[size] = tuple(sorted(partition))
    return result


def _lp_norm_words(cols, fibers, q, deep=False):
    """Restricted subgroup-fiber witnesses; not an exhaustive distance test."""
    from itertools import combinations
    found = set()
    sizes = (2, 3, 6, 9, 18) if deep else (3, 9, 18)
    for size in sizes:
        words = [fiber << (j*q)
                 for j in range(len(cols)//q) for fiber in fibers[size]]
        syndromes = []
        for word in words:
            syndrome = 0
            while word:
                bit = word & -word
                syndrome ^= cols[bit.bit_length()-1]
                word ^= bit
            syndromes.append(syndrome)
        seen = {0: 0}
        radius = 3 if deep and size == 2 else 2
        for weight in range(1, radius+1):
            for indices in combinations(range(len(words)), weight):
                syndrome, coefficient = 0, 0
                for c in indices:
                    syndrome ^= syndromes[c]
                    coefficient |= 1 << c
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = coefficient
                    continue
                dependency = coefficient ^ old
                if not dependency or size*dependency.bit_count() > 24:
                    continue
                full = 0
                while dependency:
                    bit = dependency & -dependency
                    full ^= words[bit.bit_length()-1]
                    dependency ^= bit
                if full:
                    found.add(full)
    return sorted(found, key=lambda w: (w.bit_count(), w))[:32]


def _lp_probe(cols, rng, trials, memory=(), stabilizers=None,
              split=None):
    """Retain true kernel words; logical readings exclude stabilizers.

    A high reading is a search upper estimate, never a distance certificate.
    Classical readings are only conservative component-selection heuristics.
    """
    width = len(cols)
    span = None if stabilizers is None else _row_basis(stabilizers)
    best = width+1
    retained = sorted(set(memory), key=lambda w: (w.bit_count(), w))[:48]

    def observe(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if span is None or not _in_span(word, span):
            best = word.bit_count()

    for word in retained:
        observe(word)
    order = list(range(width))
    for trial in range(trials):
        rng.shuffle(order)
        if split is not None and trial % 3 != 2:
            order.sort(key=lambda c:
                       (c < split) if trial % 3 == 0 else (c >= split))
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
            if not syndrome:
                dependencies.append(word)
                observe(word)

        raw = sorted(set(dependencies+retained),
                     key=lambda w: (w.bit_count(), w))
        words = list(raw)
        for sweep in range(2):
            reducers = words[:32]
            shortened = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                observe(word)
                shortened.append(word)
            words = sorted(set(shortened),
                           key=lambda w: (w.bit_count(), w))
        discoveries = set(words+retained)
        short = words[:32]
        for a, word in enumerate(short):
            for other in short[:a]:
                combined = word ^ other
                if combined:
                    observe(combined)
                    if combined.bit_count() <= best+4:
                        discoveries.add(combined)

        if span is not None:
            logical = [word for word in raw if not _in_span(word, span)]
            for word in logical[:4]:
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda entry: entry[0])
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:10]]
                    for a, row in enumerate(moves):
                        for other in moves[:a]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)

        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:48]
        if best <= 4:
            break
    return best, retained


def NEW_CTX(rng, G, idx):
    """One deliberate degree geometry, with fresh witness-directed filling."""
    from math import gcd
    assert G.q == 36
    ta, tb = _lp_support_catalog()[idx]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:4]), list(tb[4:])]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    q = G.q
    fibers = _lp_fibers(G)
    pools = [[g for g in G.elements if g[1] == b] for b in range(4)]
    long18 = [g for g in G.elements
              if g[1] == 2 and gcd(g[0], 9) == 1]
    long9 = [g for g in G.elements
             if g[1] == 0 and gcd(g[0], 9) == 1]
    order4 = [g for g in G.elements if g[1] % 2]
    pair_palettes = (long18, order4, long9, long18)
    triple_recipes = ((0, 1, 2), (0, 0, 1),
                      (0, 1, 3), (0, 0, 2))
    four_recipes = ((0, 1, 2, 3), (0, 0, 1, 3),
                    (0, 1, 1, 2), (0, 0, 2, 3))

    def forbidden(cell):
        # C3 cosets have fixed b and fixed a modulo three.
        occupancy = {}
        for g in cell:
            key = (g[0] % 3, g[1])
            occupancy[key] = occupancy.get(key, 0)+1
            if occupancy[key] == 3:
                return True
        return any(G.mul(G.inv(g), h) == (0, 2)
                   for a, g in enumerate(cell) for h in cell[:a])

    def draw_cell(count, model):
        if count <= 1:
            return [] if not count else [G.pick(rng)]
        if count == 2:
            base = G.pick(rng)
            relative = rng.choice(pair_palettes[model])
            return [base, G.mul(base, relative)]
        offsets = (triple_recipes if count == 3 else four_recipes)[model]
        for attempt in range(64):
            phase = rng.randrange(4)
            direction = rng.choice((-1, 1))
            cell = []
            for offset in offsets:
                b = (phase+direction*offset) % 4
                legal = [g for g in pools[b]
                         if g not in cell and not forbidden(cell+[g])]
                if not legal:
                    break
                cell.append(rng.choice(legal))
            if len(cell) == count:
                return cell
        raise ValueError("metacyclic occupancy recipe exhausted")

    def draw_grid(counts, side, model):
        return [[draw_cell(count, (model+i+int(side == "B")) % 4)
                 for count in row] for i, row in enumerate(counts)]

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", 25), 24)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                item["da"]+item["db"], -item["cycles"])

    pool, seen = [], set()

    def admit(A, B):
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (125*n <= 1000*k < 175*n):
            return
        ar, br, ac, bc = _lp_components(hx, hz, q)
        aw = _lp_norm_words(ac, fibers, q)
        bw = _lp_norm_words(bc, fibers, q)
        da, aw = _lp_probe(ac, rng, 2, aw)
        db, bw = _lp_probe(bc, rng, 2, bw)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ac, bc=bc, aw=aw, bw=bw, da=da, db=db,
                    cycles=_lp_four(ar)+_lp_four(br))
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    for attempt in range(64):
        model = (idx+attempt//16) % 4
        admit(draw_grid(sa, "A", model), draw_grid(sb, "B", model))

    mask = (1 << q)-1
    for step in range(32):
        if not pool:
            model = (idx+step) % 4
            admit(draw_grid(sa, "A", model), draw_grid(sb, "B", model))
            continue
        parent = pool[rng.randrange(min(4, len(pool)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        side = ("A" if parent["da"] < parent["db"] else "B")
        if step % 5 == 0:
            side = rng.choice(("A", "B"))
        grid, counts = (A, sa) if side == "A" else (B, sb)
        words = parent["aw"] if side == "A" else parent["bw"]
        width = len(counts[0])
        pressure = [1+sum(((word >> (j*q)) & mask).bit_count()
                          for word in words[:8]) for j in range(width)]
        j = rng.choices(range(width), weights=pressure, k=1)[0]
        occupied = [i for i in range(2) if counts[i][j]]
        i = rng.choice(occupied)
        grid[i][j] = draw_cell(counts[i][j], (idx+step) % 4)
        if step % 6 == 5:
            other = (j+1+rng.randrange(width-1)) % width
            occupied = [r for r in range(2) if counts[r][other]]
            r = rng.choice(occupied)
            grid[r][other] = draw_cell(counts[r][other], (idx+step+1) % 4)
        admit(A, B)

    if not pool:
        raise ValueError("no rank-admissible filling in cell 0.150")

    for item in pool:
        da, aw = _lp_probe(item["ac"], rng, 12, item["aw"])
        db, bw = _lp_probe(item["bc"], rng, 12, item["bw"])
        item.update(da=min(item["da"], da), db=min(item["db"], db),
                    aw=aw, bw=bw)
    pool.sort(key=merit, reverse=True)
    finalists = pool[:4]
    accepted = []

    for item in finalists:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        if _small_logical(hz, hx, n) or _small_logical(hx, hz, n):
            continue
        aw = item["aw"]+_lp_norm_words(item["ac"], fibers, q, True)
        bw = item["bw"]+_lp_norm_words(item["bc"], fibers, q, True)
        da, aw = _lp_probe(item["ac"], rng, 64, aw)
        db, bw = _lp_probe(item["bc"], rng, 64, bw)
        item.update(da=min(item["da"], da), db=min(item["db"], db),
                    aw=aw, bw=bw)

        a_seeds = [sum(((word >> (j*q)) & mask)
                       << ((j*4+beta)*q) for j in range(3))
                   for word in aw[:12] for beta in range(4)]
        b_seeds = [word << (j*4*q) for word in bw[:12] for j in range(3)]
        # Explicitly verify every embedding before full-code rescreening.
        a_seeds = [word for word in a_seeds
                   if not any((word & row).bit_count() & 1 for row in hx)]
        b_seeds = [word for word in b_seeds
                   if not any((word & row).bit_count() & 1 for row in hz)]
        dx, _ = _lp_probe(_lp_columns(hz, n), rng, 12, b_seeds,
                          stabilizers=hx, split=12*q)
        dz, _ = _lp_probe(_lp_columns(hx, n), rng, 12, a_seeds,
                          stabilizers=hz, split=12*q)
        if dx > n or dz > n:
            continue
        item["logical"] = min(dx, dz)
        accepted.append(item)

    if not accepted:
        raise ValueError("no short-logical-clean rescreened finalist")
    winner = max(accepted, key=merit)
    A, B = winner["A"], winner["B"]
    assert css_ok(winner["hx"], winner["hz"])
    assert max(row.bit_count()
               for row in winner["hx"]+winner["hz"]) <= W_MAX
    assert 125*winner["n"] <= 1000*winner["k"] < 175*winner["n"]
    assert _no_dead_lines(sa, sb)
    assert all(len(cell) <= 5 and len(cell) == len(set(cell))
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly optimized cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A large dihedral lift with legal rolled supports suffices for d30s>4: all five measured parent codes instead have d30s=4, with sector-1 witnesses.
# REFUTED: Rank surplus alone rescues the p3/p4 support pair: current-program slots 12-15 have k-kb=7..9 but d2s=12..19, below the p0/p0 measured readings of 24.
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
