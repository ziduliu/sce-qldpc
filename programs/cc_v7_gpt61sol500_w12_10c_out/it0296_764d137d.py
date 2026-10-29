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
CONCEPT = ("C23 x C2, cyclic order 46, shapes (2,3)x(2,3): even-support rank-surplus architectures with nominal A cofactor weight 32 and B weights 32..48, screened in ordinary and central-C2 coordinates, should exceed five-code fitness 60.3346 while retaining k/n<0.125.")
GROUP = ("Frozen byte-for-byte under L1. This exact C46 bed leads cell 0.100 at 60.3346, versus 54.9 on the related order-44 nonabelian bed. Order 46 maximizes the lift for these shapes. Commutativity supplies cofactor witnesses and central C2 supplies invariant coordinates; every screened syndrome and assembled rank comes from verified helpers.")
SHAPES = ("Frozen equal (2,3) sides for cell 0.100: n=598, kb=46, guaranteed rate 1/13, maximum admissible lift order 46, and equal 138-variable/92-check components. Parent slots 4,5,11 reached sector-one weight-28 minima across Acol/Bcol blocks; slots 0,6 retained weight-24 Z witnesses, and slot 0 also had mixed-sector X witnesses touching Arow0/Brow0 and Arow0/Brow1. Exact ranks retain k/n<0.125.")
FILLING = ("Twenty cards cross five even-support B architectures with four derivative-parity policies. A has counts (0,2,4;4,2,0), row totals six and column totals four; B rows total six/eight or eight/eight and columns total at most six, making both CSS row caps twelve. Every cell has zero, two or four distinct-residue terms. Nonzero derivative columns reject isolated weight-23 C23-fiber words without discarding augmentation rank surplus. Pack relative differences, seed verified cofactors, search central-C2-invariant kernels, recombine verified translations, repair actual voltages and independently rescreen logical embeddings, dressings and full-code information sets. Nominal cofactor sizes and sampled readings are heuristics, never distance certificates.")
FAMILY_NAME = "r010_C23xC2_23x23_even32_derivative_norm_logical_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Direct product C23 x C2; coprime factors make this cyclic C46.
    if not valid_q(q):
        raise ValueError("C23 x C2 has order 46")
    els = [(a, b) for b in range(2) for a in range(23)]

    def red(g):
        return (g[0] % 23, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % 23, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 23, b)

    def idx(g):
        a, b = red(g)
        return 23*b + a

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


def NEW_CTX(rng, G, idx):
    """Optimize an even-support architecture and its derivative-grade card."""
    from itertools import product
    from collections import Counter

    assert G.q == 46
    q, width = G.q, 3*G.q
    architecture, policy = divmod(idx % 20, 4)
    ta = (0, 2, 4, 4, 2, 0)
    b_cards = (
        (0, 4, 4, 4, 2, 2),  # nominal cofactor weight 48
        (2, 2, 4, 2, 4, 2),  # 44
        (0, 2, 4, 4, 4, 0),  # 40
        (0, 2, 4, 4, 2, 2),  # 36
        (2, 2, 2, 2, 2, 4),  # 32
    )
    tb = b_cards[architecture]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    # At the trivial C23 character, an even cell is epsilon times
    # the parity of its number of y terms. These small matrices choose
    # occupancy policies only; they never replace measured lifted ranks.
    derivative_plans = {}
    for counts, side in ((ta, "A"), (tb, "B")):
        options = []
        for d in product(range(2), repeat=6):
            if any(d[t] and counts[t] == 0 for t in range(6)):
                continue
            # No isolated constant-C23 word in one constituent column.
            if any(d[j] == d[j+3] == 0 for j in range(3)):
                continue
            rank = gf2_rank([
                sum(d[3*i+j] << j for j in range(3))
                for i in range(2)])
            options.append((rank, d))
        if side == "A":
            # Both end-column quartics must have odd y multiplicity.
            # Alternate between two and one primitive middle doubletons.
            wanted = 2 if policy % 2 == 0 else 1
            plans = [d for rank, d in options
                     if rank == 2 and d[1]+d[4] == wanted]
        else:
            # Lower derivative rank preserves more local rank surplus;
            # the other two policies test stronger derivative coupling.
            wanted = (min(rank for rank, d in options) if policy < 2
                      else max(rank for rank, d in options))
            plans = [d for rank, d in options if rank == wanted]
        if not plans:
            raise ValueError("derivative-grade card has no legal occupancy")
        derivative_plans[counts] = tuple(plans)

    # Projection onto C2 defines six normal-C23 fiber indicators.
    # Their actual syndromes are obtained from the verified component.
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    fibers = [sum(1 << G.idx(g) for g in pool) for pool in pools]
    fiber_words = [fiber << (j*q)
                   for j in range(3) for fiber in fibers]

    def draw_grid(counts, model):
        """Prescribed derivative grades with independently packed voltages."""
        counts = tuple(counts)
        grades = rng.choice(derivative_plans[counts])
        internal_weight, row_weight, cross_weight = (
            (2, 1, 5), (4, 2, 3), (6, 4, 1), (3, 6, 2))[policy]
        grid = [[None]*3 for _ in range(2)]
        row_used = [Counter(), Counter()]
        cross_used = Counter()
        for i in range(2):
            for j in range(3):
                t = 3*i+j
                count, grade = counts[t], grades[t]
                if not count:
                    grid[i][j] = []
                    continue
                chosen = None
                for attempt in range((1, 6, 12)[model]):
                    phase = rng.randrange(2)
                    if count == 2:
                        minority = grade
                    elif grade:
                        minority = 1
                    else:
                        minority = (2 if policy in (0, 2)
                                    else rng.choice((0, 2)))
                    quota = ([phase]*(count-minority)
                             + [1-phase]*minority)
                    rng.shuffle(quota)
                    cell = list(zip(rng.sample(range(23), count), quota))
                    ratios = [
                        G.mul(g, G.inv(h))
                        for g in cell for h in cell if g != h]
                    cross = ([] if i == 0 else [
                        G.mul(g, G.inv(h))
                        for g in grid[0][j] for h in cell])
                    penalty = (
                        internal_weight*(len(ratios)-len(set(ratios)))
                        + row_weight*sum(row_used[i][g] for g in ratios)
                        + cross_weight*(
                            len(cross)-len(set(cross))
                            + sum(cross_used[g] for g in cross)))
                    key = (penalty, rng.random())
                    if chosen is None or key < chosen[0]:
                        chosen = (key, cell, ratios, cross)
                _, cell, ratios, cross = chosen
                assert sum(g[1] for g in cell) % 2 == grade
                grid[i][j] = cell
                row_used[i].update(ratios)
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
        # Also used on the 69 central-C2-invariant coordinates.
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

    residue_mask = (1 << 23)-1

    def syndrome(cols, word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    def expand_norm2(word):
        full = 0
        for j in range(3):
            block = (word >> (23*j)) & residue_mask
            full |= (block | (block << 23)) << (q*j)
        return full

    def norm2_state(cols, ordinary):
        """Restrict verified syndromes to paired central-C2 coordinates."""
        folded = [
            cols[j*q+a] ^ cols[j*q+23+a]
            for j in range(3) for a in range(23)]
        compact = set()
        for seed in ordinary:
            norm, original, invariant = 0, 0, True
            for j in range(3):
                low = (seed >> (j*q)) & residue_mask
                high = (seed >> (j*q+23)) & residue_mask
                norm |= (low ^ high) << (23*j)
                original |= low << (23*j)
                invariant &= low == high
            coefficient = original if invariant else norm
            if coefficient and syndrome(folded, coefficient) == 0:
                compact.add(coefficient)
        return folded, sorted(compact, key=lambda w: (w.bit_count(), w))

    def refine_norm2(folded, compact, ordinary, trials):
        reading = probe(folded, compact, trials)
        physical = [expand_norm2(word) for word in compact]
        ordinary[:] = sorted(
            set(ordinary+physical),
            key=lambda w: (w.bit_count(), w))[:48]
        return 2*reading

    translation_maps = tuple(
        tuple(G.idx(G.mul(g, shift)) for g in G.elements)
        for shift in G.elements)

    def orbit_refine(cols, words):
        """Recombine true witnesses, not translated copies of cell labels."""
        cache = {}

        def images(word):
            if word in cache:
                return cache[word]
            positions, remaining = [], word
            while remaining:
                bit = remaining & -remaining
                positions.append(divmod(bit.bit_length()-1, q))
                remaining ^= bit
            valid = []
            for permutation in translation_maps:
                moved = sum(1 << (j*q+permutation[a])
                            for j, a in positions)
                if moved and syndrome(cols, moved) == 0:
                    valid.append(moved)
            assert valid, "identity translate lost a verified witness"
            cache[word] = tuple(set(valid))
            return cache[word]

        parents = sorted(
            {min(images(word)) for word in words},
            key=lambda w: (w.bit_count(), w))[:20]
        translated = {image for word in parents for image in images(word)}
        discoveries = set(parents)
        ceiling = min((w.bit_count() for w in parents),
                      default=width+1)+4
        for word in parents:
            for other in translated:
                combined = word ^ other
                if combined and combined.bit_count() <= ceiling:
                    discoveries.add(combined)
        representatives = {
            min(images(word)) for word in sorted(
                discoveries, key=lambda w: (w.bit_count(), w))[:96]}
        words[:] = sorted(
            representatives, key=lambda w: (w.bit_count(), w))[:32]
        return min((w.bit_count() for w in words), default=width+1)

    def full_reading(item):
        """Fresh full-code witnesses, classified modulo actual stabilizers."""
        n, mask = item["n"], (1 << q)-1
        best = n+1
        for side, seeds, checks, stabilizers in (
                ("A", item["aw"], item["hx"], item["hz"]),
                ("B", item["bw"], item["hz"], item["hx"])):
            span = {}
            for word in stabilizers:
                while word:
                    p = word.bit_length()-1
                    old = span.get(p)
                    if old is None:
                        span[p] = word
                        break
                    word ^= old
            ordered_span = sorted(span.items(), reverse=True)

            def logical_class(word):
                for p, row in ordered_span:
                    if (word >> p) & 1:
                        word ^= row
                return word

            cols = [0]*n
            for r, row in enumerate(checks):
                while row:
                    bit = row & -row
                    cols[bit.bit_length()-1] |= 1 << r
                    row ^= bit
            memory = []
            for seed in sorted(set(seeds), key=int.bit_count)[:12]:
                for fixed in range(3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*3*q)
                    if syndrome(cols, word):
                        continue
                    cls = logical_class(word)
                    if cls:
                        best = min(best, word.bit_count())
                        memory.append((word, cls))

            order = list(range(n))
            for trial in range(4):
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
                        cls = logical_class(word)
                        dependencies.append((word, cls))
                        if cls:
                            best = min(best, word.bit_count())

                combined = dependencies+memory
                reducers = sorted(
                    set(combined), key=lambda x: x[0].bit_count())[:48]
                reducers += [(row, 0) for row in stabilizers]
                starts = sorted(
                    {(word, cls) for word, cls in combined if cls},
                    key=lambda x: x[0].bit_count())[:12]
                discoveries = list(starts)
                for word, cls in starts:
                    for sweep in range(2):
                        weight, changed = word.bit_count(), False
                        rng.shuffle(reducers)
                        for other, oc in reducers:
                            proposal, pc = word ^ other, cls ^ oc
                            pw = proposal.bit_count()
                            if pc and pw < weight:
                                word, cls, weight = proposal, pc, pw
                                changed = True
                        if not changed:
                            break
                    best = min(best, word.bit_count())
                    discoveries.append((word, cls))
                for a, (word, cls) in enumerate(discoveries[:12]):
                    for other, oc in discoveries[:a]:
                        if cls ^ oc:
                            proposal = word ^ other
                            best = min(best, proposal.bit_count())
                memory = sorted(
                    set(discoveries),
                    key=lambda x: x[0].bit_count())[:24]
        return best

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("logical", width+1), 32)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                min(item["da"], 36)+min(item["db"], 36),
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

    finalists, seen = [], set()

    def admit(A, B):
        # Cell repairs must remain inside this card's support and grade
        # policy; copying one cell from another plan can break that policy.
        for grid, counts in ((A, ta), (B, tb)):
            if tuple(len(cell) for row in grid for cell in row) != counts:
                return
            grades = tuple(
                sum(g[1] for g in cell) % 2
                for row in grid for cell in row)
            if grades not in derivative_plans[counts]:
                return
        fingerprint = tuple(tuple(sorted(cell)) for grid in (A, B)
                            for row in grid for cell in row)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (q <= k and 8*k < n):
            return
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar), columns(br)
        aw, bw = constant_fiber_words(ac), constant_fiber_words(bc)
        aw.extend(_abelian_cofactor_seeds(G, A, ac))
        bw.extend(_abelian_cofactor_seeds(G, B, bc, inverse=True))
        an, af = norm2_state(ac, aw)
        bn, bf = norm2_state(bc, bw)
        na = refine_norm2(an, af, aw, 4)
        nb = refine_norm2(bn, bf, bw, 4)
        da = min(na, probe(ac, aw, 3))
        db = min(nb, probe(bc, bw, 3))
        item = {
            "A": A, "B": B, "k": k, "n": n, "hx": hx, "hz": hz,
            "ac": ac, "bc": bc, "aw": aw, "bw": bw,
            "an": an, "bn": bn, "af": af, "bf": bf,
            "da": da, "db": db,
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    for attempt in range(72):
        model = (idx + attempt//24) % 3
        admit(draw_grid(ta, model), draw_grid(tb, model))

    # Support architecture and derivative-grade policy remain fixed.
    # Repairs alter actual voltages in witness-heavy columns, not a
    # common automorphism or a permutation image of one labeling.
    mask = (1 << q)-1
    for step in range(48):
        if not finalists:
            admit(draw_grid(ta, step % 3), draw_grid(tb, step % 3))
            continue
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        if step % 4:
            side = "A" if parent["da"] <= parent["db"] else "B"
        else:
            side = rng.choice(("A", "B"))
        seeds = parent["aw"] if side == "A" else parent["bw"]
        seeds = sorted(seeds, key=int.bit_count)[:6]
        pressure = [
            1 + sum(((word >> (j*q)) & mask).bit_count()
                    for word in seeds)
            for j in range(3)]
        j = rng.choices(range(3), weights=pressure, k=1)[0]
        i = rng.randrange(2)
        counts = ta if side == "A" else tb
        replacement = draw_grid(counts, (idx+step//8) % 3)
        grid = A if side == "A" else B
        grid[i][j] = replacement[i][j]
        if step % 5 == 0:
            # Occasional two-cell repairs escape one-cell local optima.
            other = (j+1+rng.randrange(2)) % 3
            grid[1-i][other] = replacement[1-i][other]
        admit(A, B)

    if not finalists:
        raise ValueError("no positive-k labeling in target rate cell")
    for item in finalists:
        item["da"] = min(
            item["da"],
            refine_norm2(item["an"], item["af"], item["aw"], 16),
            probe(item["ac"], item["aw"], 16),
            orbit_refine(item["ac"], item["aw"]))
        item["db"] = min(
            item["db"],
            refine_norm2(item["bn"], item["bf"], item["bw"], 16),
            probe(item["bc"], item["bw"], 16),
            orbit_refine(item["bc"], item["bw"]))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(
            item["da"],
            refine_norm2(item["an"], item["af"], item["aw"], 64),
            probe(item["ac"], item["aw"], 96),
            orbit_refine(item["ac"], item["aw"]))
        item["db"] = min(
            item["db"],
            refine_norm2(item["bn"], item["bf"], item["bw"], 64),
            probe(item["bc"], item["bw"], 96),
            orbit_refine(item["bc"], item["bw"]))
        item["logical"] = min(dressed_reading(item), full_reading(item))
    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert q <= k and 8*k < n
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert all(
        len(cell) == len(set(cell)) <= 5
        and len(cell) in (0, 2, 4)
        and len({g[0] for g in cell}) == len(cell)
        for grid in (A, B) for row in grid for cell in row)
    for grid, counts in ((A, ta), (B, tb)):
        assert tuple(len(cell) for row in grid for cell in row) == counts
        grades = tuple(sum(g[1] for g in cell) % 2
                       for row in grid for cell in row)
        assert grades in derivative_plans[counts]
        assert all(grades[j] or grades[j+3] for j in range(3))
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the card's even-support, derivative-graded complete cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive order-46 doubletons suffice to eliminate sector-one weight-24 logicals on C46; measured current-parent slots 0 and 6 retained weight-24 Z witnesses at k=62, while slots 4,5,11 reached weight 28 at k=47,46,46.
# REFUTED: The shown row-seven norm/orbit/cofactor filling exceeds fitness 60.3346; its measured fitness was 59.4618, with five k=46 leaders retaining readings 27,28,28,28,28.
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
