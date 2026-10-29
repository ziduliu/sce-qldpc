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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): margin-preserving support transport, relative-difference packing and orbit-aware witness screening should raise five-code fitness above the bed record 54.8632 while retaining k/n<0.125.")
GROUP = ("Frozen L1 Dic11 presentation, encoding and order. RUN STATE records 54.9 on this bed versus 48.4 on the order-44 abelian bed. Right translations of the A component and left translations of the B component supply weight-preserving witness families; every translated syndrome is checked using verified component columns.")
SHAPES = ("Frozen equal (2,3) sides for cell 0.100: n=572, guaranteed k>=44 and rate>=1/13, with two length-132 components and 88 checks each. The size window permits q<=46; q=44 remains fixed. Parent slot 6 has weight-24 sector-1 Z witnesses concentrated in Bcol1/2, while slot 48 has weight-24 X witnesses concentrated in Acol1/2. Exact ranks retain k/n<0.125.")
FILLING = ("Preserve each slot's row/column term totals and degree fingerprint, but explore positive margin-equivalent cell sizes through five terms. Add a C4-occupancy model packing internal and cross-row relative differences. Retain true kernel words, shorten combinations of their verified group translates, and direct repairs towards witness-heavy columns. Independently rescreen finalists and lower selection readings using syndrome- and stabilizer-tested logical dressings. Sampled witness families and readings are not complete censuses or distance certificates.")
FAMILY_NAME = "r010_dic11_23x23_margintransport_orbitpack_dressed_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    if q != 44:
        raise ValueError("Dic11 has order 44")
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22*b + a

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
    grid = []
    involution = (11, 0)
    for i in range(2):
        row = []
        for j in range(3):
            cell = []
            while len(cell) < counts[3*i+j]:
                g = G.pick(rng)
                if g in cell:
                    continue
                if any(G.mul(G.inv(h), g) == involution for h in cell):
                    continue
                cell.append(g)
            row.append(cell)
        grid.append(row)
    return grid


def NEW_CTX(rng, G, idx):
    """Optimize one support geometry with rank-aware witness rescreening."""
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q, width = G.q, 3*G.q

    # chi(x^a y^b)=2*a+b mod 4 is the Dic11 -> C4 quotient.
    # It chooses labels and constant-fiber test vectors only. Every
    # syndrome below comes from the scaffold's verified lifted matrices.
    pools = [[] for _ in range(4)]
    for g in G.elements:
        pools[(2*g[0] + g[1]) % 4].append(g)
    fibers = [sum(1 << G.idx(g) for g in pool) for pool in pools]
    fiber_words = [fiber << (j*q)
                   for j in range(3) for fiber in fibers]
    involution = (11, 0)

    margin_cache = {}

    def transported_counts(counts):
        """Change cell sizes, not the fixed shapes or degree margins."""
        from itertools import product
        counts = tuple(counts)
        variants = margin_cache.get(counts)
        if variants is None:
            cols = tuple(counts[j]+counts[j+3] for j in range(3))
            top_sum = sum(counts[:3])
            variants = []
            for top in product(range(1, 6), repeat=3):
                bottom = tuple(cols[j]-top[j] for j in range(3))
                if sum(top) != top_sum:
                    continue
                if all(1 <= w <= 5 for w in bottom):
                    variants.append(tuple(top)+bottom)
            assert variants
            margin_cache[counts] = variants
        return rng.choice(variants)

    def draw_grid(counts, model, reprofile=True):
        if reprofile:
            counts = transported_counts(counts)

        if model == 4:
            # Pack actual group-relative labels. This construction cost
            # is only a heuristic; subsequent cycle counts use the lift.
            from collections import Counter
            recipes = {
                1: ((0,),),
                2: ((0, 1), (0, 2), (0, 3)),
                3: ((0, 1, 2), (0, 0, 1), (0, 1, 3)),
                4: ((0, 0, 1, 3), (0, 1, 1, 2), (0, 1, 2, 3)),
                5: ((0, 0, 1, 2, 3), (0, 0, 1, 1, 2),
                    (0, 1, 1, 2, 3)),
            }
            grid = [[None]*3 for _ in range(2)]
            row_used = [Counter(), Counter()]
            cross_used = Counter()
            for i in range(2):
                for j in range(3):
                    count = counts[3*i+j]
                    chosen = None
                    for attempt in range(8):
                        phase = rng.randrange(4)
                        direction = rng.choice((-1, 1))
                        offsets = rng.choice(recipes[count])
                        cell = []
                        for offset in offsets:
                            c = (phase+direction*offset) % 4
                            available = [
                                g for g in pools[c] if g not in cell
                                and all(G.mul(G.inv(h), g) != involution
                                        for h in cell)]
                            cell.append(rng.choice(available))
                        ratios = [
                            G.mul(g, G.inv(h))
                            for g in cell for h in cell if g != h]
                        cross = ([] if i == 0 else [
                            G.mul(g, G.inv(h))
                            for g in grid[0][j] for h in cell])
                        penalty = (
                            len(ratios)-len(set(ratios))
                            + sum(row_used[i][g] for g in ratios)
                            + len(cross)-len(set(cross))
                            + sum(cross_used[g] for g in cross))
                        score = (penalty, rng.random())
                        if chosen is None or score < chosen[0]:
                            chosen = (score, cell, ratios, cross)
                    _, cell, ratios, cross = chosen
                    grid[i][j] = cell
                    row_used[i].update(ratios)
                    cross_used.update(cross)
            return grid

        if model == 0:
            grid = _label_grid(rng, G, counts)
            for i in range(2):
                for j in range(3):
                    if counts[3*i+j] != 2:
                        continue
                    # Same C4 fiber means a nonidentity order-11 ratio.
                    # Other legal doubletons have relative order 4 or 22.
                    while True:
                        cell = grid[i][j]
                        relative = G.mul(G.inv(cell[0]), cell[1])
                        if relative[1] or relative[0] % 2:
                            break
                        grid[i][j] = _label_grid(rng, G, counts)[i][j]
            return grid

        long_pairs = set()
        if model == 3:
            from itertools import combinations
            positions = [(i, j) for i in range(2) for j in range(3)
                         if counts[3*i+j] == 2]
            matchings = []
            for size in range(3):
                for subset in combinations(positions, size):
                    if (len({i for i, j in subset}) == size
                            and len({j for i, j in subset}) == size):
                        matchings.append(subset)
            largest = max(map(len, matchings))
            matchings = [m for m in matchings if len(m) == largest]
            long_pairs = set(
                matchings[(idx + sum(counts)) % len(matchings)])

        grid = []
        for i in range(2):
            row = []
            for j in range(3):
                count = counts[3*i+j]
                phase = rng.randrange(4)
                if model == 1:
                    direction = rng.choice((-1, 1))
                    quota = [(phase + direction*t) % 4
                             for t in range(count)]
                elif model == 3:
                    if count == 1:
                        quota = [phase]
                    elif count == 2:
                        # Quotient difference 2: same y coset and odd
                        # rotation difference. Excluding x^11 leaves
                        # relative order 22. Odd differences give order 4.
                        delta = (2 if (i, j) in long_pairs
                                 else rng.choice((1, 3)))
                        quota = [phase, (phase+delta) % 4]
                    elif count == 3:
                        omitted = rng.randrange(4)
                        quota = [(phase+t) % 4 for t in range(4)
                                 if t != omitted]
                    else:
                        quota = [(phase+t) % 4 for t in range(count)]
                elif count == 1:
                    quota = [phase]
                elif count == 2:
                    quota = [phase, (phase + rng.choice((1, 2, 3))) % 4]
                elif count == 3:
                    quota = [phase, phase,
                             (phase + rng.choice((1, 3))) % 4]
                else:
                    quota = [phase, phase, (phase+1) % 4, (phase+2) % 4]
                cell = []
                for c in quota:
                    available = [
                        g for g in pools[c] if g not in cell
                        and all(G.mul(G.inv(h), g) != involution
                                for h in cell)
                    ]
                    cell.append(rng.choice(available))
                row.append(cell)
            grid.append(row)
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
        """Enumerate the exact kernel restricted to 12 fiber indicators."""
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
        """Actual nonzero classical kernel witnesses, not distance proofs.

        Random information sets, elementary basis shortening, all pair
        witnesses, and seeded XOR descent reduce lucky high readings.
        """
        best = min((v.bit_count() for v in structured), default=width+1)
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            discoveries = list(structured)
            pivots, basis = {}, []
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
                    basis.append(word)
                    best = min(best, word.bit_count())

            # Replace a basis vector by itself plus another basis vector.
            # Every vector remains in the verified component kernel.
            for sweep in range(2 + int(trials >= 24)):
                rng.shuffle(basis)
                changed = False
                for a in range(len(basis)):
                    word = basis[a]
                    weight = word.bit_count()
                    for b, other in enumerate(basis):
                        if a == b:
                            continue
                        trial = word ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            word, weight = trial, tw
                            changed = True
                    basis[a] = word
                    best = min(best, weight)
                if not changed:
                    break

            words = sorted(set(basis + structured), key=int.bit_count)
            seeds = list(words[:8])
            for a, word in enumerate(words):
                best = min(best, word.bit_count())
                for other in words[:a]:
                    trial = word ^ other
                    if trial:
                        tw = trial.bit_count()
                        best = min(best, tw)
                        if tw <= best+4:
                            discoveries.append(trial)
                        if len(seeds) < 16:
                            seeds.append(trial)
                        elif tw < seeds[-1].bit_count():
                            seeds[-1] = trial
                            seeds.sort(key=int.bit_count)

            # Descent uses all reducers, including structured norm words,
            # rather than only earlier entries of a sorted basis.
            rng.shuffle(words)
            for word in sorted(set(seeds), key=int.bit_count)[:8]:
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in words:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            word, weight = trial, tw
                            changed = True
                    if not changed:
                        break
                best = min(best, weight)
                discoveries.append(word)
            # Remember actual supports, not merely a scalar reading.
            structured[:] = sorted(
                set(structured+basis+discoveries),
                key=lambda v: (v.bit_count(), v))[:32]
            if best <= 20:
                break
        return best

    translations = {
        "A": [tuple(G.idx(G.mul(g, t)) for g in G.elements)
              for t in G.elements],
        "B": [tuple(G.idx(G.mul(t, g)) for g in G.elements)
              for t in G.elements],
    }

    def orbit_refine(cols, words, side):
        """Shorten true kernel words using verified group translates.

        Orbit representatives prevent one translated family from consuming
        all witness memory. The retained families are not a complete census.
        """
        cache = {}

        def images(word):
            old = cache.get(word)
            if old is not None:
                return old
            positions = []
            remaining = word
            while remaining:
                bit = remaining & -remaining
                positions.append(bit.bit_length()-1)
                remaining ^= bit
            valid = []
            for permutation in translations[side]:
                full = sum(1 << (j*q+permutation[r])
                           for c in positions
                           for j, r in (divmod(c, q),))
                syndrome = 0
                remaining = full
                while remaining:
                    bit = remaining & -remaining
                    syndrome ^= cols[bit.bit_length()-1]
                    remaining ^= bit
                if syndrome == 0:
                    valid.append(full)
            assert valid, "kernel witness lost its identity translate"
            result = tuple(set(valid))
            cache[word] = result
            return result

        memory = sorted(set(words),
                        key=lambda v: (v.bit_count(), v))[:24]
        best = min((v.bit_count() for v in memory), default=width+1)
        for sweep in range(2):
            parents = sorted({min(images(v)) for v in memory},
                             key=lambda v: (v.bit_count(), v))[:24]
            translated = set()
            for word in parents:
                translated.update(images(word))
            discoveries = set(parents)
            ceiling = best+4
            # By weight-preserving group action, one representative
            # against the other orbit covers pair weights between orbits.
            for word in parents:
                for other in translated:
                    combined = word ^ other
                    if not combined:
                        continue
                    weight = combined.bit_count()
                    best = min(best, weight)
                    if weight <= ceiling:
                        discoveries.add(combined)
            representatives = set(parents)
            for word in sorted(discoveries,
                               key=lambda v: (v.bit_count(), v))[:128]:
                representatives.add(min(images(word)))
            memory = sorted(representatives,
                            key=lambda v: (v.bit_count(), v))[:24]
        words[:] = memory
        return best

    def dressed_reading(item):
        """Find genuine logicals; adding stabilizers preserves their class."""
        hx, hz, n = item["hx"], item["hz"], item["n"]
        mask = (1 << q)-1
        best = n+1

        def basis(rows):
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

        def outside(word, pivots):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        for side, seeds, checks, stabilizers in (
                ("A", item["aw"], hx, hz),
                ("B", item["bw"], hz, hx)):
            pivots = basis(stabilizers)
            for seed in sorted(set(seeds), key=int.bit_count)[:6]:
                for fixed in range(3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*3*q)
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if not outside(word, pivots):
                        continue
                    best = min(best, word.bit_count())
                    for sweep in range(2):
                        chosen, weight = word, word.bit_count()
                        nearby = []
                        for row in stabilizers:
                            combined = word ^ row
                            cw = combined.bit_count()
                            nearby.append((cw, row))
                            if cw < weight:
                                chosen, weight = combined, cw
                        nearby.sort(key=lambda pair: pair[0])
                        moves = [row for _, row in nearby[:12]]
                        for a, row in enumerate(moves):
                            for other in moves[:a]:
                                combined = word ^ row ^ other
                                cw = combined.bit_count()
                                if cw < weight:
                                    chosen, weight = combined, cw
                        if chosen == word:
                            break
                        word = chosen
                        best = min(best, weight)
        return best

    def merit(item):
        # Classical witnesses remain conservative selection heuristics.
        # "logical" is lowered only by a verified non-stabilizer witness.
        d = min(item["da"], item["db"], item.get("logical", width+1), 32)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                min(item["da"], 36) + min(item["db"], 36),
                -item["cycles"])

    finalists, seen = [], set()

    def admit(A, B):
        ca = [[len(cell) for cell in row] for row in A]
        cb = [[len(cell) for cell in row] for row in B]
        if _grid_weights(ca, cb) > 12 or not _no_dead_lines(ca, cb):
            return
        if any(len(cell) > 5 or len(cell) != len(set(cell))
               for grid in (A, B) for row in grid for cell in row):
            return
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (q <= k and 8*k < n):
            return
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar), columns(br)
        aw, bw = constant_fiber_words(ac), constant_fiber_words(bc)
        item = {
            "A": A, "B": B, "k": k, "n": n, "hx": hx, "hz": hz,
            "ac": ac, "bc": bc, "aw": aw, "bw": bw,
            "da": probe(ac, aw, 4), "db": probe(bc, bw, 4),
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[10:]

    for attempt in range(80):
        # Five occupancy models, with independently transported margins.
        model = (idx + attempt//16) % 5
        admit(draw_grid(ta, model), draw_grid(tb, model))

    for step in range(40):
        if not finalists:
            model = (idx + step) % 5
            admit(draw_grid(ta, model), draw_grid(tb, model))
            continue
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]

        # Change the limiting component preferentially, but retain
        # exploration when the two readings tie or every fifth step.
        if step % 5 == 0 or parent["da"] == parent["db"]:
            side = rng.choice(("A", "B"))
        else:
            side = "A" if parent["da"] < parent["db"] else "B"
        grid = A if side == "A" else B
        counts = tuple(len(cell) for row in grid for cell in row)
        model = (idx+step) % 5
        if step % 8 == 7:
            # Replace the whole component when transporting term counts:
            # copying only one changed-size cell would break its margins.
            proposal = draw_grid(counts, model, reprofile=True)
            if side == "A":
                A = proposal
            else:
                B = proposal
        else:
            proposal = draw_grid(counts, model, reprofile=False)
            seeds = parent["aw"] if side == "A" else parent["bw"]
            mask = (1 << q)-1
            pressure = [
                1 + sum(((word >> (j*q)) & mask).bit_count()
                        for word in seeds[:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = proposal[i][j]
            if step % 6 == 5:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = proposal[1-i][other]
        admit(A, B)

    if not finalists:
        raise ValueError("no positive-k labeling in target rate cell")

    # Fresh information sets retain supports from earlier searches.
    # Orbit recombination supplements, rather than replaces, rescreening.
    for item in finalists:
        item["da"] = min(
            item["da"], probe(item["ac"], item["aw"], 16),
            orbit_refine(item["ac"], item["aw"], "A"))
        item["db"] = min(
            item["db"], probe(item["bc"], item["bw"], 16),
            orbit_refine(item["bc"], item["bw"], "B"))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(
            item["da"], probe(item["ac"], item["aw"], 64),
            orbit_refine(item["ac"], item["aw"], "A"))
        item["db"] = min(
            item["db"], probe(item["bc"], item["bw"], 64),
            orbit_refine(item["bc"], item["bw"], "B"))
        item["logical"] = dressed_reading(item)

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert q <= k and 8*k < n
    assert _grid_weights(
        [[len(cell) for cell in row] for row in A],
        [[len(cell) for cell in row] for row in B]) <= 12
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the adaptively optimized full cell without subsequent redraw."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rank surplus alone predicts stronger distance readings: parent slot 0 had k=60 and d2s=22, while slots 11/22/32 had k=44 and d30s=26.
# REFUTED: Matching-controlled order-22 doubletons and independent scalar component rescreening remove the weight-24 bottleneck; the measured current parent still has d30s=24 at k=56 in slot 6 and k=52 in slot 48, with sector-1 minima.
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
