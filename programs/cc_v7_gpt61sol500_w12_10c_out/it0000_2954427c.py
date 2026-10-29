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
CONCEPT = ("C9 x C5, cyclic order 45, shapes (2,3)x(2,3): matched rank-one augmentation/GF4 reductions, cancellation-free cofactors and independently deepened constituent/full-logical searches should supply five fresh k=57 codes with measured readings >=26, exceeding fitness 64.6322.")
GROUP = ("L3 jump from C46 to genuinely different C45, encoded as C9 x C5. The measured C45 bed reached 59.2 and included k=53,d30s=26; superiority is not assumed. Odd order makes F2[C45] semisimple, and projection onto C3 supplies a dimension-two GF4 constituent. All matrix actions, restricted syndromes and ranks come from verified construction helpers.")
SHAPES = ("Keep equal (2,3) sides for cell 0.100: n=585, kb=45, guaranteed rate 1/13, and equal 135-variable/90-check constituents. These shapes allow order 46, while the new family fixes 45. The current parent's principal minima occupied sector-one Acol/Bcol blocks; slot 3 also had a weight-27 mixed-sector Z logical touching Arow0/Brow0 and Arow1/Brow0. Exact assembled ranks enforce k=57 and k/n<0.125.")
FILLING = ("Five order-9/order-45 architectures cross four intrinsic trinomial occupancy policies. Counts [[2,2,2],[2,2,3]] give row totals 6/7 and column totals 4/4/5, so both check maxima are twelve. Prescribe each GF4 reduction as a nonzero outer product, require verified constituent rank 87, and reject actual cofactor cancellations below 28. Search actual C3/C5/C9/C15 indicator kernels, translated witnesses and ordinary kernels; repair individual voltages while preserving GF4 sums. Equally deepen finalists, test exact full logical classes and stabilizer dressings, and conservatively reject repeated uncoloured Tanner invariants. Found weights remain upper witnesses, never distance certificates.")
FAMILY_NAME = "r010_C9xC5_23x23_GF4rank1_ordercards_norm13515_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 45

def custom_ops(q):
    # Coprime cyclic factors C9 x C5 give cyclic C45.
    # Projection a mod 3 supplies the C3/GF4 constituent.
    if not valid_q(q):
        raise ValueError("C9 x C5 has order 45")
    els = [(a, b) for b in range(5) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 5)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % 9, (b + d) % 5)

    def inv(g):
        a, b = red(g)
        return ((-a) % 9, (-b) % 5)

    def idx(g):
        a, b = red(g)
        return 9*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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


def _legacy_C46_CTX(rng, G, idx):
    """Inactive historical C46 optimizer; NEW_CTX below replaces it."""
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


_C45_RECIPES = (
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
_C45_CARDS = ((0, 0), (0, 3), (0, 6), (3, 6), (6, 8))
_C45_COUNTS = ((2, 2, 2), (2, 2, 3))


def _c45_columns(rows, width):
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _c45_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _c45_basis(rows):
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


def _c45_outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


@lru_cache(maxsize=1)
def _c45_atlas(G):
    """Compile only scaffold-generated single-term constituent actions."""
    assert G.q == 45 and A_SHAPE == B_SHAPE == (2, 3)
    q, width = G.q, 3*G.q
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[] for _ in range(3)] for _ in range(2)]
            B = [[[] for _ in range(3)] for _ in range(2)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            cols = _c45_columns(ar if side == "A" else br, width)
            block = tuple(cols[:q])
            assert all(value.bit_count() == 1 for value in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    partitions = {}
    for size, generator in (
            (3, (3, 0)), (5, (0, 1)),
            (9, (1, 0)), (15, (3, 1))):
        unseen, fibers = set(range(q)), []
        while unseen:
            g = G.elements[min(unseen)]
            h, orbit = g, []
            for _ in range(size):
                orbit.append(G.idx(h))
                h = G.mul(h, generator)
            assert h == g and len(set(orbit)) == size
            unseen.difference_update(orbit)
            fibers.append(sum(1 << t for t in orbit))
        assert len(fibers)*size == q
        partitions[size] = tuple(sorted(fibers))

    translations = tuple(
        tuple(G.idx(G.mul(g, shift)) for g in G.elements)
        for shift in G.elements)
    return atlas, partitions, translations


def _c45_compile(G, atlas, side, grid):
    q = G.q
    cols = [0]*(3*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                for h, value in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+h] ^= value << (i*q)
    rows = [0]*(2*q)
    for c, original in enumerate(cols):
        value = original
        while value:
            bit = value & -value
            rows[bit.bit_length()-1] |= 1 << c
            value ^= bit
    return rows, cols


def _c45_draw(rng, G, recipe, mode):
    """GF4 outer-product sums with prescribed relative orders.

    GF4 values of the C3 cosets are 1,2,3. Two complementary
    cosets evaluate to the omitted coset. A trinomial uses that
    coset plus a repeated coset, so it has the same GF4 value.
    """
    from collections import Counter

    top, bottom = recipe
    uf = [rng.randrange(3) for _ in range(2)]
    vf = [rng.randrange(3) for _ in range(3)]
    grid = [[None]*3 for _ in range(2)]
    internal = [Counter(), Counter()]
    crossing = Counter()

    for i in range(2):
        for j in range(3):
            count = _C45_COUNTS[i][j]
            exponent = (uf[i]+vf[j]) % 3
            if count == 2:
                quota = [(exponent+1) % 3, (exponent+2) % 3]
            else:
                repeated = (
                    exponent if mode == 0
                    else (exponent+1) % 3 if mode == 1
                    else (exponent+2) % 3 if mode == 2
                    else (exponent+rng.randrange(3)) % 3)
                quota = [exponent, repeated, repeated]

            winner = None
            for _ in range(12):
                rng.shuffle(quota)
                residues = []
                for coset in quota:
                    legal = [a for a in range(9)
                             if a % 3 == coset and a not in residues]
                    residues.append(rng.choice(legal))
                phase = rng.randrange(5)
                if count == 2:
                    grade = top[j] if i == 0 else bottom[j]
                    grades = [phase, (phase+rng.randrange(1, 5)) % 5
                              if grade else phase]
                elif mode == 0:
                    grades = rng.sample(range(5), 3)
                elif mode == 1:
                    grades = [phase, phase, (phase+1) % 5]
                    rng.shuffle(grades)
                elif mode == 2:
                    grades = [rng.randrange(5) for _ in range(3)]
                else:
                    grades = [(phase+t) % 5 for t in (0, 1, 3)]
                    rng.shuffle(grades)
                cell = list(zip(residues, grades))
                differences = [
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h]
                cross = ([] if i == 0 else [
                    G.mul(g, G.inv(h))
                    for g in grid[0][j] for h in cell])
                penalty = (
                    4*(len(differences)-len(set(differences)))
                    + 2*sum(internal[i][d] for d in differences)
                    + 4*(len(cross)-len(set(cross)))
                    + 3*sum(crossing[d] for d in cross))
                key = (penalty, rng.random())
                if winner is None or key < winner[0]:
                    winner = (key, cell, differences, cross)
            _, cell, differences, cross = winner
            value = 0
            for a, b in cell:
                value ^= (1, 2, 3)[a % 3]
            assert value == (1, 2, 3)[exponent]
            assert len(cell) == len(set(cell)) == count
            grid[i][j] = cell
            internal[i].update(differences)
            crossing.update(cross)
    return grid


def _c45_probe(cols, rng, trials, seeds=(), stabilizers=None):
    """Actual kernel/logical witnesses, never a distance certificate."""
    width = len(cols)
    span = None if stabilizers is None else _c45_basis(stabilizers)
    memory = sorted(
        {w for w in seeds if w and _c45_syndrome(cols, w) == 0},
        key=lambda w: (w.bit_count(), w))[:40]
    best = width+1
    logical_memory = set()

    def observe(word):
        nonlocal best
        if not word:
            return
        if span is None or _c45_outside(word, span):
            best = min(best, word.bit_count())
            if span is not None:
                logical_memory.add(word)

    for word in memory:
        observe(word)
    order = list(range(width))
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
                observe(word)

        words = sorted(set(memory+dependencies),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        discoveries.update(words[:48])
        for sweep in range(2):
            donors = sorted(set(words+memory),
                            key=lambda w: (w.bit_count(), w))[:32]
            rng.shuffle(donors)
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
                observe(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
            discoveries.update(words[:48])

        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        if span is not None:
            # Logical directions are retained separately from short checks.
            starts = sorted(logical_memory,
                            key=lambda w: (w.bit_count(), w))[:12]
            donors = short[:24]
            for word in starts:
                for sweep in range(2):
                    weight, chosen = word.bit_count(), word
                    for donor in donors:
                        candidate = word ^ donor
                        cw = candidate.bit_count()
                        if (0 < cw < weight
                                and _c45_outside(candidate, span)):
                            chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                for sweep in range(3):
                    changed = False
                    for row in stabilizers:
                        candidate = word ^ row
                        if candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
                observe(word)
                discoveries.add(word)
            logical_memory = set(sorted(
                logical_memory, key=lambda w: (w.bit_count(), w))[:20])

        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:40]
        if best <= 4:
            break
    if span is not None:
        memory = sorted(set(memory) | logical_memory,
                        key=lambda w: (w.bit_count(), w))
    assert all(_c45_syndrome(cols, word) == 0 for word in memory)
    return best, list(memory)


def _c45_restricted(cols, physical, rng, trials, seeds=(), exact=False):
    """Search disjoint subgroup indicators using verified syndromes."""
    folded = [_c45_syndrome(cols, word) for word in physical]
    if exact:
        pivots, dependencies = {}, []
        for c, syn in enumerate(folded):
            word = 1 << c
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
        combinations = [0]
        for word in dependencies:
            combinations += [old ^ word for old in combinations]
        compact = sorted((w for w in combinations if w),
                         key=lambda w: (w.bit_count(), w))[:40]
    else:
        _, compact = _c45_probe(folded, rng, trials, seeds)

    expanded = []
    for coefficient in compact:
        word = 0
        while coefficient:
            bit = coefficient & -coefficient
            word ^= physical[bit.bit_length()-1]
            coefficient ^= bit
        if word:
            assert _c45_syndrome(cols, word) == 0
            expanded.append(word)
    return compact, expanded


def _c45_orbits(G, cols, words, maps):
    """Recombine translated witnesses, not translated candidate labelings."""
    q = G.q
    discoveries = set(words)
    parents = sorted(discoveries,
                     key=lambda w: (w.bit_count(), w))[:8]
    orbit_representatives, translated = set(), set()
    for seed in parents:
        positions, remaining = [], seed
        while remaining:
            bit = remaining & -remaining
            positions.append(divmod(bit.bit_length()-1, q))
            remaining ^= bit
        orbit = set()
        for permutation in maps:
            moved = sum(1 << (j*q+permutation[h])
                        for j, h in positions)
            if moved and _c45_syndrome(cols, moved) == 0:
                orbit.add(moved)
        assert orbit
        representative = min(orbit)
        if representative not in orbit_representatives:
            orbit_representatives.add(representative)
            translated.update(orbit)
    ceiling = min((w.bit_count() for w in words), default=len(cols)+1)+4
    for word in orbit_representatives:
        for other in translated:
            candidate = word ^ other
            if candidate and candidate.bit_count() <= ceiling:
                discoveries.add(candidate)
    return sorted(discoveries,
                  key=lambda w: (w.bit_count(), w))[:40]


def _c45_refine(G, item, partitions, maps, rng, trials, orbit=False):
    q = G.q
    words = list(item["words"])
    for size, fibers in partitions.items():
        physical = [fiber << (j*q) for j in range(3) for fiber in fibers]
        compact, expanded = _c45_restricted(
            item["cols"], physical, rng,
            max(3, trials//2), item["compact"].get(size, ()),
            exact=size in (9, 15))
        item["compact"][size] = compact
        words.extend(expanded)
    if orbit:
        words = _c45_orbits(G, item["cols"], words, maps)
    reading, words = _c45_probe(item["cols"], rng, trials, words)
    item["d"] = min(item["d"], reading)
    item["words"] = words


def _c45_component_pool(G, atlas, partitions, maps, rng,
                        side, recipe, mode, batch):
    """Independent GF4-compatible voltage optimization."""
    pool, seen = [], set()

    def merit(item):
        d = min(item["d"], 28)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"])

    def admit(grid):
        labels = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if labels in seen:
            return
        seen.add(labels)
        rows, cols = _c45_compile(G, atlas, side, grid)
        # One augmentation and two GF4 dimensions are rank deficits.
        # Require the actual binary rank; do not infer full-code k here.
        if gf2_rank(rows) != 2*G.q-3:
            return
        cofactors = _abelian_cofactor_seeds(
            G, grid, cols, inverse=side == "B")
        if not cofactors or min(w.bit_count() for w in cofactors) != 28:
            return
        item = dict(
            grid=grid, cols=cols, rows=rows, words=cofactors,
            compact={}, d=28, cycles=_four_cycles(rows))
        _c45_refine(G, item, partitions, maps, rng, 3)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    for attempt in range(192+64*batch):
        admit(_c45_draw(rng, G, recipe, mode))

    mask = (1 << G.q)-1
    for step in range(40):
        if not pool:
            admit(_c45_draw(rng, G, recipe, mode))
            continue
        parent = rng.choice(pool[:min(4, len(pool))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        pressure = [
            1+sum(((word >> (j*G.q)) & mask).bit_count()
                  for word in parent["words"][:8])
            for j in range(3)]
        j = rng.choices(range(3), weights=pressure, k=1)[0]
        i = rng.randrange(2)
        cell = grid[i][j]
        for _ in range(2 if step % 5 == 4 else 1):
            position = rng.randrange(len(cell))
            a, b = cell[position]
            aa = (a+3*rng.choice((1, 2))) % 9
            bb = b
            if len(cell) == 3 and step % 3 == 0:
                bb = rng.randrange(5)
            elif len(cell) == 2:
                grade = recipe[0][j] if i == 0 else recipe[1][j]
                if grade and step % 3 == 0:
                    other = cell[1-position][1]
                    bb = rng.choice([t for t in range(5) if t != other])
            candidate = (aa, bb)
            if candidate not in cell:
                cell[position] = candidate
        # Individual a changes by multiples of three preserve GF4 sums.
        admit(grid)

    for item in pool:
        _c45_refine(G, item, partitions, maps, rng, 16, orbit=True)
    pool.sort(key=merit, reverse=True)
    pool = pool[:4]
    for item in pool:
        _c45_refine(G, item, partitions, maps, rng, 64, orbit=True)
    return sorted(pool, key=merit, reverse=True)[:3]


def _c45_signature(hx, hz, n):
    """Uncoloured Tanner invariant; equality is rejected conservatively."""
    from collections import Counter

    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    links = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    keys = []
    for i, neighbors in enumerate(links):
        local = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        keys.append((weights[i], tuple(sorted(local.items()))))
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
    incident = [[] for _ in range(n)]
    for weight, original in zip(weights, rows):
        word = original
        while word:
            bit = word & -word
            incident[bit.bit_length()-1].append(weight)
            word ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(values)) for values in incident).items())))


@lru_cache(maxsize=1)
def _c45_bank(G):
    """Pure deterministic bank of distinct, independently optimized cards."""
    from itertools import combinations_with_replacement

    assert G.q == 45 and A_SHAPE == B_SHAPE == (2, 3)
    atlas, partitions, maps = _c45_atlas(G)
    q, width = G.q, 3*G.q
    mask = (1 << q)-1
    assert _grid_weights(_C45_COUNTS, _C45_COUNTS) == 12
    assert _no_dead_lines(_C45_COUNTS, _C45_COUNTS)
    extra_cards = tuple(combinations_with_replacement(range(9), 2))
    bank, emitted = [], set()

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("logical", item["n"]+1), 28)
        load = sum(w.bit_count() <= d+1
                   for component in (item["a"], item["b"])
                   for w in component["words"])
        return (item["k"]*d*d/item["n"], d, -load,
                -item["a"]["cycles"]-item["b"]["cycles"])

    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        lane = slot//20
        card = (_C45_CARDS[geometry] if lane == 0 else
                extra_cards[(5*lane+7*geometry) % len(extra_cards)])
        recipes = (_C45_RECIPES[card[0]], _C45_RECIPES[card[1]])
        local = random.Random(
            BASE_SEED*100003 + 45010357 + 104729*slot)
        selected = None

        for batch in range(2):
            ap = _c45_component_pool(
                G, atlas, partitions, maps, local,
                "A", recipes[0], mode, batch)
            bp = _c45_component_pool(
                G, atlas, partitions, maps, local,
                "B", recipes[1], mode, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (n == 585 and k == 57 and 8*k < n):
                        continue
                    ar, br = _component_checks(hx, hz, q)
                    assert _c45_columns(ar, width) == a["cols"]
                    assert _c45_columns(br, width) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            proposals.sort(key=merit, reverse=True)

            finalists = []
            for item in proposals:
                signature = _c45_signature(item["hx"], item["hz"], item["n"])
                if signature in emitted:
                    continue
                a_seeds = [
                    sum(((word >> (j*q)) & mask)
                        << ((3*j+fixed)*q) for j in range(3))
                    for word in item["a"]["words"][:24]
                    for fixed in range(3)]
                b_seeds = [
                    word << (fixed*3*q)
                    for word in item["b"]["words"][:24]
                    for fixed in range(3)]
                assert all(_c45_syndrome(
                    _c45_columns(item["hx"], item["n"]), w) == 0
                    for w in a_seeds)
                assert all(_c45_syndrome(
                    _c45_columns(item["hz"], item["n"]), w) == 0
                    for w in b_seeds)
                dx, xwords = _c45_probe(
                    _c45_columns(item["hz"], item["n"]),
                    local, 4, b_seeds, item["hx"])
                dz, zwords = _c45_probe(
                    _c45_columns(item["hx"], item["n"]),
                    local, 4, a_seeds, item["hz"])
                if dx > item["n"] or dz > item["n"]:
                    continue
                item.update(
                    logical=min(dx, dz), xwords=xwords, zwords=zwords,
                    signature=signature)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # Every finalist receives the same fresh deeper full-code budget.
            for item in finalists:
                dx, _ = _c45_probe(
                    _c45_columns(item["hz"], item["n"]), local, 8,
                    item["xwords"], item["hx"])
                dz, _ = _c45_probe(
                    _c45_columns(item["hx"], item["n"]), local, 8,
                    item["zwords"], item["hz"])
                item["logical"] = min(item["logical"], dx, dz)

            if finalists:
                winner = max(finalists, key=merit)
                A, B = winner["A"], winner["B"]
                assert css_ok(winner["hx"], winner["hz"])
                assert max(row.bit_count()
                           for row in winner["hx"]+winner["hz"]) == 12
                assert all(
                    len(cell) == len(set(cell)) == _C45_COUNTS[i][j]
                    for grid in (A, B)
                    for i, row in enumerate(grid)
                    for j, cell in enumerate(row))
                for recipe, grid in zip(recipes, (A, B)):
                    top, bottom = recipe
                    for i in range(2):
                        for j in range(3):
                            cell = grid[i][j]
                            if len(cell) == 2:
                                grade = top[j] if i == 0 else bottom[j]
                                assert int(cell[0][1] != cell[1][1]) == grade
                                assert cell[0][0] % 3 != cell[1][0] % 3
                emitted.add(winner["signature"])
                selected = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in grid)
                    for grid in (A, B))
                break

        # Failed cards never substitute an emitted or transported code.
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the fresh order-45 GF4/rank-screened structural task."""
    item = _c45_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh C45 GF4-rank-one filling passed this card")
    A, B = item
    return {
        "A": [[list(cell) for cell in row] for row in A],
        "B": [[list(cell) for cell in row] for row in B]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without additional label draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive order-46 doubletons plus commuting-cofactor and ordinary component rescreening eliminate C46 sector-one weight-24 logicals; measured current-parent slots 0 and 6 retained weight-24 Z witnesses at k=62, while slots 4,5,11 reached weight 28 at k=47,46,46.
# REFUTED: Distinct central-grade cards automatically yield nonisomorphic C46 codes; the current table collapses slots 17 and 18 onto slots 5 and 4, with further isomorphism duplicates in slots 28,35,41,42,48,49,58.
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
