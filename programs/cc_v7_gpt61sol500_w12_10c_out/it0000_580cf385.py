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
CONCEPT = ("C17 x C17, order 289, shapes (1,1)x(1,1): independent five-term polynomials sharing two or three Frobenius character orbits should exceed fitness 64.6322 without the rectangular weight-28 cofactor bottleneck; exact assembled ranks must retain 0<k/n<0.125.")
GROUP = ("L2 order repair within the unchanged two-cyclic-factor direct-product rule. The new size factor two makes the parent's order 46 inadmissible at n=92. Choose C17 x C17: order 289 is the largest equal-factor square order below 300, giving n=578. Since ord_17(2)=8, a prescribed nontrivial character zero forces an eight-dimensional constituent deficit. The related order-288 square direct-product bed scored 53.3, not evidence that this new bed will win. All matrix actions and ranks use verified helpers.")
SHAPES = ("Change both sides to (1,1) for cell 0.100: n=578, guaranteed kb=0, and equal square 289-variable/289-check constituent maps. Positive k and k/n<0.125 are exact full-rank gates, not consequences of shape counting. Parent weight-28 minima mainly occupied sector-one Acol/Bcol blocks; slot 3 also had mixed-sector weight-27 Z witnesses. Square shapes remove the rectangular three-column cofactor construction, while full logical screening still examines both sectors and types.")
FILLING = ("Twenty tasks cross five slopes with four common/asymmetric two- or three-character prescriptions. GF256 pair/triple syndrome matching constructs distinct five-term supports, with collinear supports excluded. An empty three-character library drops only its extra character, never substitutes an emitted code. Both CSS check weights are ten. Independently optimize fresh support pairs, seed both sectors from verified constituent kernels, retain exact nonzero logical classes, equally deepen finalists and reject repeated uncoloured Tanner invariants. Character constraints and found logical weights are not distance certificates.")
FAMILY_NAME = "r010_directproduct_C17xC17_11x11_character23_retainedlogical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 289

def custom_ops(q):
    # Same two-cyclic-factor direct-product rule, now C17 x C17.
    if not valid_q(q):
        raise ValueError("C17 x C17 has order 289")
    m, r = 17, 17
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, (-b) % r)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 1)               # (ra, ca)
B_SHAPE = (1, 1)               # (rb, cb)


def _support_designs():
    """Canonical supports, interleaved across augmentation-rank strata."""
    from itertools import product, permutations
    perms = list(permutations(range(3)))
    profiles = []
    for t in product(range(1, 5), repeat=6):
        rows = (sum(t[:3]), sum(t[3:]))
        cols = tuple(t[j] + t[j+3] for j in range(3))
        if min(rows) < 5 or max(rows) > 8:
            continue
        if min(cols) < 3 or max(cols) > 6:
            continue
        p0 = sum((t[j] & 1) << j for j in range(3))
        p1 = sum((t[j+3] & 1) << j for j in range(3))
        rank = int(bool(p0 or p1))
        rank += int(bool(p0 and p1 and p0 != p1))
        images = []
        for order in ((0, 1), (1, 0)):
            for perm in perms:
                images.append(tuple(t[3*r+c] for r in order for c in perm))
        if t == min(images):
            profiles.append((t, rows, cols, rank))

    buckets = {}
    for a, pa in enumerate(profiles):
        ta, ar, ac, ka = pa
        for tb, br, bc, kb in profiles[a:]:
            if max(ar) + max(bc) > 12 or max(br) + max(ac) > 12:
                continue
            checks = tuple(sorted([x+y for x in ar for y in bc] +
                                  [x+y for x in br for y in ac]))
            qubits = tuple(sorted([x+y for x in ac for y in bc] +
                                  [x+y for x in ar for y in br]))
            priority = (min(ac+bc), min(ar+br), sum(ta)+sum(tb),
                        -max(ac+bc), -max(ar+br))
            stratum = tuple(sorted((ka, kb)))
            buckets.setdefault(stratum, []).append(
                (priority, (checks, qubits), ta, tb))
    for bucket in buckets.values():
        bucket.sort(reverse=True)

    # Rank-deficient designs receive slots, rather than being discarded
    # before their actual lifted ranks and witnesses can be measured.
    strata = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
    positions = {s: 0 for s in strata}
    seen = set()
    designs = []
    while True:
        added = False
        for stratum in strata:
            bucket = buckets.get(stratum, ())
            while positions[stratum] < len(bucket):
                item = bucket[positions[stratum]]
                positions[stratum] += 1
                _, fingerprint, ta, tb = item
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                designs.append((ta, tb))
                added = True
                break
        if not added:
            break
    if len(designs) < 20:
        raise ValueError("fewer than twenty distinct support designs")
    return tuple(designs)


_SUPPORT_DESIGNS = _support_designs()


def _four_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for a, x in enumerate(bits):
            for y in bits[a+1:]:
                key = (x, y)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_probe(rows, width, rng):
    """Upper readings on classical kernel distance, used only as heuristics."""
    best = width + 1
    for _ in range(4):
        perm = list(range(width))
        rng.shuffle(perm)
        pivots = {}
        for original in rows:
            row = 0
            while original:
                bit = original & -original
                row ^= 1 << perm[bit.bit_length()-1]
                original ^= bit
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for f in range(width):
            if f in pivots:
                continue
            v = 1 << f
            for p, row in ordered:
                if (row & v).bit_count() & 1:
                    v ^= 1 << p
            best = min(best, v.bit_count())
            basis.append(v)
        for a, v in enumerate(basis):
            for u in basis[a+1:]:
                best = min(best, (v ^ u).bit_count())
    return best


def _component_checks(hx, hz, q):
    """Extract the two component actions from the VERIFIED LP lift."""
    mask = (1 << q)-1
    arows = []
    for i in range(2):
        for r in range(q):
            source = hx[(i*3)*q+r]
            packed = 0
            for j in range(3):
                packed |= ((source >> (j*3*q)) & mask) << (j*q)
            arows.append(packed)
    bmask = (1 << (3*q))-1
    brows = [hz[a*q+r] & bmask for a in range(2) for r in range(q)]
    return arows, brows


def _label_grid(rng, G, counts):
    """Distinct residues; doubletons have primitive order-46 differences."""
    grid = []
    for i in range(2):
        row = []
        for j in range(3):
            count = counts[3*i+j]
            residues = rng.sample(range(23), count)
            parity = [rng.randrange(2) for _ in range(count)]
            if count == 2:
                parity[1] = 1 - parity[0]
            row.append(list(zip(residues, parity)))
        grid.append(row)
    return grid


def _abelian_cofactor_seeds(G, grid, cols, inverse=False):
    """Group-algebra minors, accepted only after verified syndrome tests.

    Commutativity supplies a constructive constituent-kernel direction.
    This computes words, not an alternative parity-matrix lift.
    """
    q = G.q
    entries = [
        [set(G.inv(g) if inverse else g for g in cell) for cell in row]
        for row in grid
    ]

    def product(left, right):
        result = set()
        for g in left:
            for h in right:
                term = G.mul(g, h)
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
        return result

    minors = []
    for omitted in range(3):
        s, t = [j for j in range(3) if j != omitted]
        minors.append(
            product(entries[0][s], entries[1][t])
            ^ product(entries[0][t], entries[1][s]))

    seeds = []
    for shift in G.elements[::7]:
        word = 0
        for j, minor in enumerate(minors):
            for g in minor:
                word ^= 1 << (j*q + G.idx(G.mul(g, shift)))
        if not word:
            continue
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        if syndrome == 0:
            seeds.append(word)
    return seeds


def _retired_rectangular_ctx(rng, G, idx):
    """Inactive rectangular optimizer; the square policy below replaces it."""
    assert G.q == 46
    q, width = G.q, 3*G.q
    ta = tb = (2, 2, 2, 2, 2, 3)
    sa = sb = [[2, 2, 2], [2, 2, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    # Each recipe gives the relative C2 grades of the three top
    # doubletons and the two bottom doubletons. A zero grade means
    # relative order 23, a one grade means relative order 46.
    # In columns zero/one, at least one row has grade one: otherwise
    # a C23-coset indicator would give an immediate weight-23 kernel.
    recipes = (
        ((1, 1, 1), (0, 0)),
        ((1, 1, 1), (0, 1)),
        ((1, 1, 1), (1, 1)),
        ((1, 1, 0), (0, 0)),
        ((1, 1, 0), (0, 1)),
        ((1, 1, 0), (1, 1)),
        ((0, 1, 1), (1, 0)),
        ((0, 1, 1), (1, 1)),
        ((0, 1, 0), (1, 0)),
    )
    cards = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (1, 1),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (1, 7), (1, 8), (2, 3), (2, 6), (3, 6),
    )
    card = cards[idx % 20]
    prescribed = {"A": recipes[card[0]], "B": recipes[card[1]]}
    assert all(
        top[0] or top[1]
        for top, bottom in prescribed.values())
    assert all(
        top[j] or bottom[j]
        for top, bottom in prescribed.values() for j in range(2))

    # Projection onto C2 defines six normal-C23 fiber indicators.
    # Their actual syndromes are obtained from the verified component.
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    fibers = [sum(1 << G.idx(g) for g in pool) for pool in pools]
    fiber_words = [fiber << (j*q)
                   for j in range(3) for fiber in fibers]

    def draw_grid(counts, model, side):
        """Pack fresh labels while preserving the card's intrinsic grades."""
        from collections import Counter

        top, bottom = prescribed[side]
        grid = [[None]*3 for _ in range(2)]
        internal = [Counter(), Counter()]
        cross_used = Counter()
        coefficients = ((4, 2, 4), (6, 4, 1), (2, 3, 6))[model % 3]
        for i in range(2):
            for j in range(3):
                count = counts[3*i+j]
                chosen = None
                for attempt in range(12):
                    phase = rng.randrange(2)
                    if count == 2:
                        grade = top[j] if i == 0 else bottom[j]
                        quota = [phase, phase ^ grade]
                    elif model == 1:
                        # Homogeneous and mixed trinomials are genuinely
                        # different occupancies, not common translations.
                        quota = [phase]*3
                    else:
                        quota = [phase, phase, 1-phase]
                    rng.shuffle(quota)
                    residues = rng.sample(range(23), count)
                    cell = list(zip(residues, quota))
                    differences = [
                        G.mul(g, G.inv(h))
                        for g in cell for h in cell if g != h]
                    cross = ([] if i == 0 else [
                        G.mul(g, G.inv(h))
                        for g in grid[0][j] for h in cell])
                    a, b, c = coefficients
                    penalty = (
                        a*(len(differences)-len(set(differences)))
                        + b*sum(internal[i][d] for d in differences)
                        + c*(len(cross)-len(set(cross))
                             + sum(cross_used[d] for d in cross)))
                    key = (penalty, rng.random())
                    if chosen is None or key < chosen[0]:
                        chosen = (key, cell, differences, cross)
                _, cell, differences, cross = chosen
                grid[i][j] = cell
                internal[i].update(differences)
                cross_used.update(cross)
        return grid

    def columns(rows):
        out = [0]*width
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                lo = row & -row
                out[lo.bit_length()-1] |= bit
                row ^= lo
        return out

    def constant_fiber_words(cols):
        """Enumerate the exact kernel restricted to six C23 indicators."""
        small_cols = []
        for word in fiber_words:
            syn = 0
            while word:
                lo = word & -word
                syn ^= cols[lo.bit_length()-1]
                word ^= lo
            small_cols.append(syn)
        pivots = {}
        deps = []
        for c, syn in enumerate(small_cols):
            word = 1 << c
            while syn:
                p = syn.bit_length()-1
                if p not in pivots:
                    pivots[p] = (syn, word)
                    break
                ps, pw = pivots[p]
                syn ^= ps
                word ^= pw
            if not syn:
                deps.append(word)
        words = [0]
        for dep in deps:
            words += [word ^ dep for word in words]
        if len(words) == 1:
            return []
        words = sorted(words[1:], key=lambda v: (v.bit_count(), v))[:12]
        expanded = []
        for word in words:
            full = 0
            while word:
                lo = word & -word
                full ^= fiber_words[lo.bit_length()-1]
                word ^= lo
            expanded.append(full)
        return expanded

    def probe(cols, structured, trials):
        """Actual kernel witnesses, with memory local to this candidate.

        The mutable list retains short words between independent
        information sets. Every retained word is a nonzero true kernel
        word; a high reading is never a distance certificate.
        """
        width = len(cols)
        memory = list(set(structured))
        best = min((v.bit_count() for v in memory), default=width+1)
        order = list(range(width))
        for trial in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = list(memory)
            for c in order:
                syn, word = cols[c], 1 << c
                while syn:
                    p = syn.bit_length()-1
                    hit = pivots.get(p)
                    if hit is None:
                        pivots[p] = (syn, word)
                        break
                    syn ^= hit[0]
                    word ^= hit[1]
                if not syn:
                    words.append(word)
                    best = min(best, word.bit_count())

            words = sorted(set(words), key=lambda v: (v.bit_count(), v))
            discoveries = []
            for sweep in range(1 + int(trials >= 8)):
                changed = False
                words.sort(key=lambda v: (v.bit_count(), v))
                for a in range(len(words)):
                    word, weight = words[a], words[a].bit_count()
                    for reducer in words[:a]:
                        combined = word ^ reducer
                        cw = combined.bit_count()
                        if combined and cw < weight:
                            word, weight = combined, cw
                            changed = True
                    words[a] = word
                    best = min(best, weight)
                if not changed:
                    break

            # Cross-information-set recombination supplements basis
            # descent, which can otherwise discard useful short words.
            short = sorted(set(words+memory),
                           key=lambda v: (v.bit_count(), v))[:40]
            for a, word in enumerate(short):
                for other in short[:a]:
                    combined = word ^ other
                    if not combined:
                        continue
                    cw = combined.bit_count()
                    best = min(best, cw)
                    if cw <= best+4:
                        discoveries.append(combined)
            if trials >= 8 and trial % 4 == 0:
                seeds = short[:12]
                for a, word in enumerate(seeds):
                    for b in range(a):
                        pair = word ^ seeds[b]
                        for c in range(b):
                            combined = pair ^ seeds[c]
                            if not combined:
                                continue
                            cw = combined.bit_count()
                            best = min(best, cw)
                            if cw <= best+4:
                                discoveries.append(combined)
            memory = sorted(set(memory+words+discoveries),
                            key=lambda v: (v.bit_count(), v))[:24]
            if best <= 4:
                break
        structured[:] = memory
        return best

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    def dressed_reading(item):
        """Search actual logical representatives near component witnesses.

        Embeddings are syndrome-checked on the verified full matrices.
        Stabilizer membership is tested exactly before dressing. Adding
        stabilizers preserves the logical class and can reach both sectors.
        """
        hx, hz, n = item["hx"], item["hz"], item["n"]

        def row_basis(rows):
            basis = {}
            for word in rows:
                while word:
                    p = word.bit_length()-1
                    hit = basis.get(p)
                    if hit is None:
                        basis[p] = word
                        break
                    word ^= hit
            return basis

        def outside(word, basis):
            while word:
                p = word.bit_length()-1
                hit = basis.get(p)
                if hit is None:
                    return True
                word ^= hit
            return False

        xb, zb = row_basis(hx), row_basis(hz)
        mask = (1 << q)-1
        best = n+1
        ceiling = min(item["da"], item["db"])+4
        for side, seeds, checks, stabilizers, basis in (
                ("A", item["aw"], hx, hz, zb),
                ("B", item["bw"], hz, hx, xb)):
            for seed in sorted(set(seeds), key=int.bit_count)[:8]:
                if seed.bit_count() > ceiling:
                    continue
                for fixed in range(3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((j*3+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*3*q)
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if not outside(word, basis):
                        continue
                    best = min(best, word.bit_count())
                    for sweep in range(3):
                        chosen, weight = word, word.bit_count()
                        nearby = []
                        for row in stabilizers:
                            combined = word ^ row
                            cw = combined.bit_count()
                            nearby.append((cw, row))
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                        # Pair dressings can cross a barrier where neither
                        # individual check decreases the representative.
                        nearby.sort(key=lambda entry: entry[0])
                        rows = [row for _, row in nearby[:16]]
                        for a, row in enumerate(rows):
                            for other in rows[:a]:
                                combined = word ^ row ^ other
                                cw = combined.bit_count()
                                if 0 < cw < weight:
                                    chosen, weight = combined, cw
                        if chosen == word:
                            break
                        word = chosen
                        best = min(best, weight)
        return best

    mask = (1 << q)-1
    residue_mask = (1 << 23)-1

    def syndrome(cols, word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    # Compile single-term actions ONLY from the verified scaffold.
    # Subsequent constituent compilation is their F2-linear superposition.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[] for _ in range(3)] for _ in range(2)]
            B = [[[] for _ in range(3)] for _ in range(2)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            block = tuple(columns(ar if side == "A" else br)[:q])
            assert all(v.bit_count() == 1 for v in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    contribution = atlas[side][G.idx(g)]
                    for h, value in enumerate(contribution):
                        cols[j*q+h] ^= value << (i*q)
        return cols

    def component_rows(cols):
        rows = [0]*(2*q)
        for c, value in enumerate(cols):
            while value:
                bit = value & -value
                rows[bit.bit_length()-1] |= 1 << c
                value ^= bit
        return rows

    def expand_norm(word):
        physical = 0
        for j in range(3):
            block = (word >> (23*j)) & residue_mask
            physical |= (block | (block << 23)) << (q*j)
        return physical

    def norm_state(cols, seeds):
        """Restrict actual syndromes to central-C2 invariant coordinates."""
        folded = [
            cols[j*q+a] ^ cols[j*q+23+a]
            for j in range(3) for a in range(23)]
        memory = set()
        for seed in seeds:
            norm, compressed, invariant = 0, 0, True
            for j in range(3):
                low = (seed >> (j*q)) & residue_mask
                high = (seed >> (j*q+23)) & residue_mask
                norm |= (low ^ high) << (23*j)
                compressed |= low << (23*j)
                invariant &= low == high
            word = compressed if invariant else norm
            if word and syndrome(folded, word) == 0:
                memory.add(word)
        return folded, sorted(memory, key=lambda w: (w.bit_count(), w))

    translation_maps = tuple(
        tuple(G.idx(G.mul(g, shift)) for g in G.elements)
        for shift in G.elements)

    def orbit_reading(cols, words):
        """Shorten verified witness orbits, never relabel candidate cells."""
        cache = {}

        def images(word):
            old = cache.get(word)
            if old is not None:
                return old
            positions, remaining = [], word
            while remaining:
                bit = remaining & -remaining
                positions.append(divmod(bit.bit_length()-1, q))
                remaining ^= bit
            valid = set()
            for permutation in translation_maps:
                moved = sum(
                    1 << (j*q+permutation[h]) for j, h in positions)
                if moved and syndrome(cols, moved) == 0:
                    valid.add(moved)
            assert valid
            result = tuple(valid)
            cache[word] = result
            return result

        representatives = {
            min(images(w)) for w in sorted(
                set(words), key=lambda w: (w.bit_count(), w))[:24]}
        parents = sorted(
            representatives, key=lambda w: (w.bit_count(), w))[:8]
        translated = set()
        for word in parents:
            translated.update(images(word))
        discoveries = set(words)
        ceiling = min((w.bit_count() for w in words), default=width+1)+4
        for word in parents:
            for other in translated:
                combined = word ^ other
                if combined and combined.bit_count() <= ceiling:
                    discoveries.add(combined)
        words[:] = sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:32]
        return min((w.bit_count() for w in words), default=width+1)

    def refine_component(item, norm_trials, ordinary_trials, orbit=False):
        nd = 2*probe(item["folded"], item["norm_words"], norm_trials)
        item["words"].extend(
            expand_norm(w) for w in item["norm_words"])
        item["words"][:] = sorted(
            set(item["words"]), key=lambda w: (w.bit_count(), w))[:48]
        od = probe(item["cols"], item["words"], ordinary_trials)
        item["d"] = min(item["d"], nd, od)
        if orbit:
            item["d"] = min(
                item["d"], orbit_reading(item["cols"], item["words"]))
        assert all(syndrome(item["cols"], w) == 0
                   for w in item["words"])

    def component_merit(item):
        d = min(item["d"], 28)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"])

    def component_pool(side):
        pool, seen = [], set()
        counts = ta if side == "A" else tb

        def admit(grid):
            labels = tuple(tuple(sorted(cell))
                           for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            cols = compile_component(side, grid)
            # This is a verified constituent rank, not an assumed
            # assembled homology formula.
            if gf2_rank(cols) != 2*q-1:
                return
            cofactors = _abelian_cofactor_seeds(
                G, grid, cols, inverse=(side == "B"))
            if not cofactors:
                return
            # Nominal minors have weights 8,10,10. Reject actual
            # cancellations immediately rather than hoping a short
            # randomized search misses the resulting lighter witness.
            if min(w.bit_count() for w in cofactors) != 28:
                return
            words = cofactors + constant_fiber_words(cols)
            folded, norm_words = norm_state(cols, words)
            rows = component_rows(cols)
            item = dict(
                grid=grid, cols=cols, words=words,
                folded=folded, norm_words=norm_words,
                d=28, cycles=_four_cycles(rows))
            refine_component(item, 6, 3)
            pool.append(item)
            pool.sort(key=component_merit, reverse=True)
            del pool[10:]

        for attempt in range(240):
            model = (idx+attempt//80+int(side == "B")) % 3
            admit(draw_grid(counts, model, side))

        # Repair actual voltages within the fixed grade card.
        for step in range(40):
            if not pool:
                admit(draw_grid(counts, step % 3, side))
                continue
            parent = rng.choice(pool[:min(4, len(pool))])
            grid = [[list(cell) for cell in row]
                    for row in parent["grid"]]
            replacement = draw_grid(counts, (idx+step) % 3, side)
            pressure = [
                1+sum(((w >> (j*q)) & mask).bit_count()
                      for w in parent["words"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = replacement[i][j]
            if step % 5 == 4:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = replacement[1-i][other]
            admit(grid)

        for item in pool:
            refine_component(item, 32, 24, orbit=True)
        pool.sort(key=component_merit, reverse=True)
        pool = pool[:5]
        for item in pool:
            refine_component(item, 128, 96, orbit=True)
        pool.sort(key=component_merit, reverse=True)
        return pool[:3]

    ap, bp = component_pool("A"), component_pool("B")
    proposals = []
    for a in ap:
        for b in bp:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (q <= k and 8*k < n):
                continue
            ar, br = _component_checks(hx, hz, q)
            # Validate the compiled actions against every complete lift
            # that reaches assembly.
            assert columns(ar) == a["cols"]
            assert columns(br) == b["cols"]
            proposals.append(dict(
                A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                ac=a["cols"], bc=b["cols"],
                aw=list(a["words"]), bw=list(b["words"]),
                da=a["d"], db=b["d"],
                cycles=a["cycles"]+b["cycles"]))

    if not proposals:
        raise ValueError("no rank-legal independent grade-card assembly")
    finalists = sorted(proposals, key=merit, reverse=True)[:4]

    def full_logical_reading(checks, stabilizers, n, trials):
        """Fresh mixed-sector witnesses; only non-stabilizers count."""
        cols = [0]*n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= 1 << r
                row ^= bit
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = span.get(p)
                if old is None:
                    span[p] = row
                    break
                row ^= old

        def outside(word):
            while word:
                p = word.bit_length()-1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        best = n+1
        order = list(range(n))
        for trial in range(trials):
            rng.shuffle(order)
            pivots, dependencies = {}, []
            for c in order:
                syn, word = cols[c], 1 << c
                while syn:
                    p = syn.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syn, word)
                        break
                    syn ^= old[0]
                    word ^= old[1]
                if not syn:
                    dependencies.append(word)
            # Keep logical directions separately from short stabilizers.
            logicals = sorted(
                (w for w in dependencies if outside(w)),
                key=int.bit_count)[:12]
            reducers = sorted(dependencies, key=int.bit_count)[:40]
            for seed in logicals:
                word = seed
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    for reducer in reducers:
                        combined = word ^ reducer
                        cw = combined.bit_count()
                        if 0 < cw < weight and outside(combined):
                            chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                for sweep in range(2):
                    changed = False
                    for row in stabilizers:
                        combined = word ^ row
                        if combined.bit_count() < word.bit_count():
                            word = combined
                            changed = True
                    if not changed:
                        break
                assert syndrome(cols, word) == 0 and outside(word)
                best = min(best, word.bit_count())
        return best

    for item in finalists:
        item["da"] = min(
            item["da"], probe(item["ac"], item["aw"], 32))
        item["db"] = min(
            item["db"], probe(item["bc"], item["bw"], 32))
        item["logical"] = dressed_reading(item)
        item["logical"] = min(
            item["logical"],
            full_logical_reading(item["hx"], item["hz"], item["n"], 4),
            full_logical_reading(item["hz"], item["hx"], item["n"], 4))

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 598 and q <= k and 8*k < n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    actual_a = [[len(cell) for cell in row] for row in A]
    actual_b = [[len(cell) for cell in row] for row in B]
    assert actual_a == actual_b == [[2, 2, 2], [2, 2, 3]]
    assert _grid_weights(actual_a, actual_b) == 12
    assert _no_dead_lines(actual_a, actual_b)
    assert all(
        len(cell) == len(set(cell)) <= 5
        and len({g[0] for g in cell}) == len(cell)
        for grid in (A, B) for row in grid for cell in row)
    for side, grid in (("A", A), ("B", B)):
        top, bottom = prescribed[side]
        assert all(
            (grid[0][j][0][1] ^ grid[0][j][1][1]) == top[j]
            for j in range(3))
        assert all(
            (grid[1][j][0][1] ^ grid[1][j][1][1]) == bottom[j]
            for j in range(2))
    return {"A": A, "B": B}


from functools import lru_cache


def _d17_columns(rows, width):
    cols = [0]*width
    for r, original in enumerate(rows):
        word = original
        while word:
            bit = word & -word
            cols[bit.bit_length()-1] |= 1 << r
            word ^= bit
    return tuple(cols)


def _d17_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _d17_basis(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return tuple(sorted(span.items(), reverse=True))


def _d17_class(word, span):
    for p, row in span:
        if (word >> p) & 1:
            word ^= row
    return word


def _d17_dependencies(cols, local):
    """True kernel dependencies of the supplied verified syndromes."""
    order = list(range(len(cols)))
    local.shuffle(order)
    pivots, words = {}, []
    for c in order:
        syn, word = cols[c], 1 << c
        while syn:
            p = syn.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = (syn, word)
                break
            syn ^= old[0]
            word ^= old[1]
        if not syn:
            words.append(word)
    return words


@lru_cache(maxsize=32)
def _d17_library(characters):
    """Five-support character zeros, not an alternative parity lift.

    GF256 uses z^8+z^4+z^3+z^2+1. Its element 2 is primitive;
    alpha=2^15 has order 17. Each byte records one character value.
    Pair/triple matches exhaust supports until the storage quota fills.
    """
    from itertools import combinations

    def mul(a, b):
        value = 0
        while b:
            if b & 1:
                value ^= a
            b >>= 1
            a <<= 1
            if a & 256:
                a ^= 0x11d
        return value

    alpha = 1
    for _ in range(15):
        alpha = mul(alpha, 2)
    powers = [1]
    for _ in range(16):
        powers.append(mul(powers[-1], alpha))
    assert mul(powers[-1], alpha) == 1 and alpha != 1

    coordinates = [(a, b) for b in range(17) for a in range(17)]
    values = [
        sum(powers[(u*a+v*b) % 17] << (8*t)
            for t, (u, v) in enumerate(characters))
        for a, b in coordinates]
    pairs = {}
    for i in range(289):
        for j in range(i):
            pairs.setdefault(values[i] ^ values[j], []).append((j, i))

    found = set()
    for i, j, h in combinations(range(289), 3):
        matches = pairs.get(values[i] ^ values[j] ^ values[h], ())
        for a, b in matches:
            support = (i, j, h, a, b)
            if len(set(support)) != 5:
                continue
            points = [coordinates[t] for t in support]
            x0, y0 = points[0]
            x1, y1 = points[1]
            dx, dy = (x1-x0) % 17, (y1-y0) % 17
            if all((dx*(y-y0)-dy*(x-x0)) % 17 == 0
                   for x, y in points[2:]):
                # Exclude one-dimensional cell polynomials.
                continue
            # Remove independent cell translations before optimization.
            canonical = min(
                tuple(sorted((x-u) % 17 + 17*((y-v) % 17)
                             for x, y in points))
                for u, v in points)
            found.add(canonical)
            if len(found) >= 384:
                return tuple(sorted(found))
    return tuple(sorted(found))


@lru_cache(maxsize=1)
def _d17_atlas(G):
    """Both square actions compiled only from verified unit-cell lifts."""
    assert G.q == 289 and A_SHAPE == B_SHAPE == (1, 1)
    q, mask = G.q, (1 << G.q)-1
    atlas = {}
    for side in ("A", "B"):
        entries = []
        for g in G.elements:
            A, B = [[[]]], [[[]]]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 2*q
            if side == "A":
                first = tuple(row & mask for row in hx)
                second = tuple(row >> q for row in hz)
            else:
                first = tuple(row >> q for row in hx)
                second = tuple(row & mask for row in hz)
            assert all(row.bit_count() == 1
                       for row in first+second)
            entries.append((first, second))
        atlas[side] = tuple(entries)
    return atlas


@lru_cache(maxsize=1024)
def _d17_component(G, side, support):
    """F2 superposition of scaffold-generated unit actions."""
    q = G.q
    first, second = [0]*q, [0]*q
    atlas = _d17_atlas(G)[side]
    for index in support:
        u, v = atlas[index]
        for r in range(q):
            first[r] ^= u[r]
            second[r] ^= v[r]
    assert all(row.bit_count() == 5 for row in first+second)
    return tuple(first), tuple(second)


def _d17_seeds(rows, local):
    cols = _d17_columns(rows, 289)
    words = set()
    for _ in range(2):
        words.update(_d17_dependencies(cols, local))
    return tuple(sorted(words, key=lambda w: (w.bit_count(), w))[:20])


def _d17_probe(checks, stabilizers, width, local, trials, seeds=()):
    """Actual logical witnesses modulo the exact stabilizer span.

    Nonzero quotient directions are retained separately from short
    stabilizers. No no-witness sentinel earns selection credit.
    """
    cols = _d17_columns(checks, width)
    span = _d17_basis(stabilizers)
    retained, memory = {}, {}
    best = width+1

    def observe(word, cls):
        nonlocal best
        if word and cls:
            retained[word] = cls
            best = min(best, word.bit_count())

    for word in seeds:
        if word and _d17_syndrome(cols, word) == 0:
            cls = _d17_class(word, span)
            memory[word] = cls
            observe(word, cls)

    for trial in range(trials):
        dependencies = _d17_dependencies(cols, local)
        directions = dict(memory)
        directions.update(retained)
        for word in dependencies:
            cls = _d17_class(word, span)
            directions[word] = cls
            observe(word, cls)

        short = sorted(directions.items(),
                       key=lambda item: (item[0].bit_count(), item[0]))[:48]
        starts = sorted(retained.items(),
                        key=lambda item: (item[0].bit_count(), item[0]))[:12]
        reducers = short + [(row, 0) for row in stabilizers]
        local.shuffle(reducers)
        for word, cls in starts:
            weight = word.bit_count()
            for sweep in range(3):
                changed = False
                for donor, dc in reducers:
                    candidate, cc = word ^ donor, cls ^ dc
                    cw = candidate.bit_count()
                    if cc and cw < weight:
                        word, cls, weight = candidate, cc, cw
                        changed = True
                if not changed:
                    break
            observe(word, cls)

            # A pair of checks can cross a one-check descent barrier.
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda item: item[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:12]]
                for a, row in enumerate(moves):
                    for other in moves[:a]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
                observe(word, cls)

        logicals = sorted(retained.items(),
                          key=lambda item: (item[0].bit_count(), item[0]))[:12]
        for a, (word, cls) in enumerate(logicals):
            for other, oc in logicals[:a]:
                observe(word ^ other, cls ^ oc)
        retained = dict(sorted(
            retained.items(),
            key=lambda item: (item[0].bit_count(), item[0]))[:24])
        memory = dict(short[:24])
        memory.update(retained)
        if best <= 4:
            break

    assert retained, "positive k must supply a nonzero logical direction"
    assert all(_d17_syndrome(cols, word) == 0
               and _d17_class(word, span) == cls
               for word, cls in retained.items())
    return best, tuple(sorted(retained,
                              key=lambda w: (w.bit_count(), w))[:16])


def _d17_signature(hx, hz, n):
    """Uncoloured Tanner invariant; unequal signatures prove nonisomorphism."""
    from collections import Counter

    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0]*len(rows)
    links = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                links[i][j] = links[j][i] = overlap
    keys = []
    for i, neighbors in enumerate(links):
        profile = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        keys.append((weights[i], tuple(sorted(profile.items()))))
    history = []
    for _ in range(2):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: t for t, key in enumerate(sorted(histogram))}
        colors = [palette[key] for key in keys]
        keys = [
            (colors[i], tuple(sorted(Counter(
                (colors[j], overlap)
                for j, overlap in neighbors.items()).items())))
            for i, neighbors in enumerate(links)]
    return (n, len(rows), tuple(history),
            tuple(sorted(Counter(
                c.bit_count() for c in _d17_columns(rows, n)).items())))


@lru_cache(maxsize=1)
def _d17_bank(G):
    """Pure deterministic optimization of twenty character architectures."""
    assert G.q == 289 and A_SHAPE == B_SHAPE == (1, 1)
    q, n = G.q, 2*G.q
    slopes = (1, 2, 3, 5, 7)
    base = ((1, 0), (0, 1))
    bank, emitted = [], set()

    def merit(item):
        d = min(item["dx"], item["dz"], 64)
        return (item["k"]*d*d/n, d, -item["cycles"])

    for slot in range(N_SLOTS):
        geometry, policy = divmod(slot % 20, 4)
        t = (slopes[geometry]+3*(slot//20)-1) % 16+1
        extra = (1, t)
        ca = base + ((extra,) if policy in (1, 2) else ())
        cb = base + ((extra,) if policy in (1, 3) else ())
        alib, blib = _d17_library(ca), _d17_library(cb)
        # Explicit relaxation of an exhausted extra-character task.
        # The common two-character requirement is never dropped.
        if not alib:
            ca, alib = base, _d17_library(base)
        if not blib:
            cb, blib = base, _d17_library(base)
        if not alib or not blib:
            bank.append(None)
            continue
        local = random.Random(
            BASE_SEED*100003+28901073+104729*slot)
        raw, labels = [], set()

        for attempt in range(96):
            a, b = local.choice(alib), local.choice(blib)
            key = (a, b)
            if key in labels:
                continue
            labels.add(key)
            ax, az = _d17_component(G, "A", a)
            bx, bz = _d17_component(G, "B", b)
            hx = tuple(x | (y << q) for x, y in zip(ax, bx))
            hz = tuple(x | (y << q) for x, y in zip(bz, az))
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (0 < k and 8*k < n):
                continue
            cycles = _four_cycles(ax)+_four_cycles(bz)
            item = dict(
                a=a, b=b, ax=ax, az=az, bx=bx, bz=bz,
                hx=hx, hz=hz, k=k, cycles=cycles)
            raw.append(item)
            # Preserve several exact rank strata before witness screening.
            raw.sort(key=lambda x: (x["k"], -x["cycles"]), reverse=True)
            buckets = {}
            for candidate in raw:
                buckets.setdefault(candidate["k"], []).append(candidate)
            raw = [candidate for bucket in buckets.values()
                   for candidate in bucket[:3]]
            raw.sort(key=lambda x: (x["k"], -x["cycles"]), reverse=True)
            del raw[12:]

        for item in raw:
            xseeds = (_d17_seeds(item["bz"], local)
                      + tuple(w << q for w in
                              _d17_seeds(item["az"], local)))
            zseeds = (_d17_seeds(item["ax"], local)
                      + tuple(w << q for w in
                              _d17_seeds(item["bx"], local)))
            dx, wx = _d17_probe(
                item["hz"], item["hx"], n, local, 3, xseeds)
            dz, wz = _d17_probe(
                item["hx"], item["hz"], n, local, 3, zseeds)
            item.update(dx=dx, dz=dz, wx=wx, wz=wz)

        finalists = sorted(raw, key=merit, reverse=True)[:4]
        # Identical fresh full-code budgets for all finalists.
        for item in finalists:
            dx, wx = _d17_probe(
                item["hz"], item["hx"], n, local, 12, item["wx"])
            dz, wz = _d17_probe(
                item["hx"], item["hz"], n, local, 12, item["wz"])
            item.update(
                dx=min(item["dx"], dx), dz=min(item["dz"], dz),
                wx=wx, wz=wz)

        selected = None
        for item in sorted(finalists, key=merit, reverse=True):
            signature = _d17_signature(item["hx"], item["hz"], n)
            if signature in emitted:
                continue
            A = [[[G.elements[g] for g in item["a"]]]]
            B = [[[G.elements[g] for g in item["b"]]]]
            vx, vz, vn = lifted_matrices(G, A, B)
            assert tuple(vx) == item["hx"] and tuple(vz) == item["hz"]
            assert vn == n == 578
            assert lifted_k(G, A, B) == item["k"]
            assert 0 < item["k"] and 8*item["k"] < n
            assert css_ok(vx, vz)
            assert _grid_weights([[5]], [[5]]) == 10
            assert _no_dead_lines([[5]], [[5]])
            assert max(row.bit_count() for row in vx+vz) == 10
            assert len(set(A[0][0])) == len(set(B[0][0])) == 5
            emitted.add(signature)
            selected = (
                tuple(A[0][0]), tuple(B[0][0]),
                item["k"], merit(item), (ca, cb))
            break
        # No orbit-copy or previously emitted-code fallback.
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this slot's fresh, rank-gated square character task."""
    item = _d17_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh square character filling passed this task")
    a, b, k, selection, characters = item
    return {"A": [[list(a)]], "B": [[list(b)]],
            "k": k, "selection": selection, "characters": characters}


def CELL(rng, G, side, i, j, ctx):
    """Return every complete cell of the selected square filling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive order-46 doubletons plus commuting-cofactor and ordinary component rescreening eliminate C46 sector-one weight-24 logicals; measured current-parent slots 0 and 6 retained weight-24 Z witnesses at k=62, while slots 4,5,11 reached weight 28 at k=47,46,46.
# REFUTED: Distinct central-grade cards automatically give nonisomorphic fillings; the measured rank-91 parent retained 51 of 60 slots and collapsed nine as isomorphic duplicates.
# REFUTED: Weight-28 cofactors and the parent's full-code screening exclude lighter mixed-sector logicals; measured parent slot 3 had a weight-27 Z witness touching Arow0/Brow0 and Arow1/Brow0.
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
