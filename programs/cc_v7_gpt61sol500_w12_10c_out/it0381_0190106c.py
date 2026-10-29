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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): central-involution kernel screening and independently selected constituents should raise five-code fitness above 54.8632 while keeping every emitted rate below 0.125.")
GROUP = ("Frozen L1 Dic11 presentation, order and encoding. This bed has reached 56.9 in RUN STATE; the current program scores 54.8632. The normal C11 subgroup supplies quotient indicators, while the central involution supplies a separate 66-coordinate invariant constituent subspace. All actions, restricted syndromes, assembled ranks and logical classes use verified construction helpers.")
SHAPES = ("Frozen equal (2,3) sides for cell 0.100: n=572, kb=44, guaranteed rate 1/13, and two equal 132-variable/88-check constituents. The size window permits q<=46, but Dic11 fixes q=44. Parent slots 0 and 7 retained sector-one weight-24 minima across Acol/Bcol blocks; slot 0 also had mixed-sector X witnesses touching Arow0/Brow0 or Arow0/Brow1. Slot 5 reached weight 27 at k=44. Exact assembled ranks enforce k/n<0.125.")
FILLING = ("Twenty deliberate cards cross five support pairs with four intrinsic C4 occupancy policies, including order-11, order-22 and order-four doubletons. Independently optimize each constituent using normal-C11 indicators, central-involution invariant kernels, ordinary information sets and syndrome-verified translated-witness recombination. Repair actual witness-heavy cells, deepen retained components, assemble rate-legal pairs and search genuine full-code logical classes with class-preserving stabilizer dressings. Both row-weight caps remain twelve. Rank surplus is rewarded only alongside surviving witness readings; sampled readings are upper witnesses, not distance certificates.")
FAMILY_NAME = "r010_dic11_23x23_centralnorm_independent_quotientcards_v4"

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
    """Independently optimize one support/quotient architecture."""
    assert G.q == 44
    assert A_SHAPE == B_SHAPE == (2, 3)
    rng = random.Random(rng.getrandbits(64) ^ 0xD114466)
    geometry, policy = divmod(idx % 20, 4)
    p = (2, 2, 2, 2, 2, 3)
    q7 = (2, 2, 3, 2, 3, 2)
    r = (1, 2, 3, 4, 1, 2)
    s = (1, 2, 4, 3, 3, 1)
    t = (1, 3, 3, 3, 2, 2)
    plans = ((p, p), (p, q7), (q7, q7), (q7, r), (s, t))
    ta, tb = plans[geometry]
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

    def draw_grid(counts, model, side):
        """Prescribe intrinsic quotient differences, then pack fresh labels.

        Quotient gaps zero, two and odd give doubleton relative orders
        eleven, twenty-two and four respectively, after excluding z.
        Relative-difference packing is heuristic; actual cycles and
        every witness syndrome are measured on scaffold-derived actions.
        """
        from collections import Counter
        grid = [[None]*3 for _ in range(2)]
        internal = [Counter(), Counter()]
        crossings = Counter()
        coefficients = ((4, 2, 3), (2, 4, 5), (6, 3, 1))[model % 3]
        for i in range(2):
            for j in range(3):
                count = counts[3*i+j]
                if count == 1:
                    offsets = (0,)
                elif count == 2:
                    if policy == 0:
                        gap = 1
                    elif policy == 1:
                        gap = 2 if i == 0 else 1
                    elif policy == 2:
                        # Complementary homogeneous/mixed doubletons:
                        # a doubleton-only column is never homogeneous
                        # in both rows.
                        gap = (0 if j % 2 == 0 else 1) if i == 0 else (
                            1 if j % 2 == 0 else 2)
                    else:
                        gap = 1 if i == 0 else 2
                    offsets = (0, gap)
                elif count == 3:
                    offsets = ((0, 1, 2), (0, 1, 3),
                               (0, 0, 1), (0, 1, 3))[policy]
                else:
                    offsets = ((0, 1, 2, 3), (0, 0, 1, 3),
                               (0, 1, 1, 2), (0, 0, 2, 3))[policy]

                chosen = None
                for attempt in range(10 + 4*(model % 3)):
                    phase = rng.randrange(4)
                    direction = rng.choice((-1, 1))
                    quota = [(phase+direction*v) % 4 for v in offsets]
                    rng.shuffle(quota)
                    cell = []
                    for c in quota:
                        available = [
                            g for g in pools[c] if g not in cell
                            and all(G.mul(G.inv(h), g) != involution
                                    for h in cell)]
                        if not available:
                            break
                        cell.append(rng.choice(available))
                    if len(cell) != count:
                        continue
                    if side == "A":
                        ratios = [G.mul(g, G.inv(h))
                                  for g in cell for h in cell if g != h]
                        cross = ([] if i == 0 else [
                            G.mul(g, G.inv(h))
                            for g in grid[0][j] for h in cell])
                    else:
                        ratios = [G.mul(G.inv(g), h)
                                  for g in cell for h in cell if g != h]
                        cross = ([] if i == 0 else [
                            G.mul(G.inv(g), h)
                            for g in grid[0][j] for h in cell])
                    a, b, c = coefficients
                    penalty = (
                        a*(len(ratios)-len(set(ratios)))
                        + b*sum(internal[i][v] for v in ratios)
                        + c*(len(cross)-len(set(cross))
                             + sum(crossings[v] for v in cross)))
                    key = (penalty, rng.random())
                    if chosen is None or key < chosen[0]:
                        chosen = (key, cell, ratios, cross)
                if chosen is None:
                    raise ValueError("Dic11 quotient occupancy exhausted")
                _, cell, ratios, cross = chosen
                grid[i][j] = cell
                internal[i].update(ratios)
                crossings.update(cross)
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

    def syndrome(cols, word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    # Compile regular actions exclusively from scaffold-generated units.
    # Compilation below is their F2-linear superposition, not another lift.
    atlas = {}
    zero = [[[] for _ in range(3)] for _ in range(2)]
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(3)] for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero) if side == "A" else (zero, unit)
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
                    for h, value in enumerate(atlas[side][G.idx(g)]):
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

    # z=(11,0) pairs a with a+11 inside EACH b-coset.
    # These are physical indicators; their syndromes come from cols.
    pair_words = [
        ((1 << (j*q+22*b+a)) | (1 << (j*q+22*b+a+11)))
        for j in range(3) for b in range(2) for a in range(11)]

    def expand_central(word):
        physical = 0
        while word:
            bit = word & -word
            physical ^= pair_words[bit.bit_length()-1]
            word ^= bit
        return physical

    def central_state(cols, seeds):
        folded = [syndrome(cols, word) for word in pair_words]
        memory = set()
        for seed in seeds:
            trace, compressed, invariant = 0, 0, True
            for j in range(3):
                for b in range(2):
                    low = (seed >> (j*q+22*b)) & 2047
                    high = (seed >> (j*q+22*b+11)) & 2047
                    shift = 22*j+11*b
                    trace |= (low ^ high) << shift
                    compressed |= low << shift
                    invariant &= low == high
            word = compressed if invariant else trace
            if word and syndrome(folded, word) == 0:
                memory.add(word)
        return folded, sorted(memory, key=lambda w: (w.bit_count(), w))

    # A kernels admit right translations, B kernels left translations.
    # Every translated witness is nevertheless syndrome-tested.
    translations = {
        side: tuple(
            tuple(G.idx(G.mul(g, shift) if side == "A"
                        else G.mul(shift, g)) for g in G.elements)
            for shift in G.elements)
        for side in ("A", "B")}

    def orbit_refine(item):
        parents = sorted(set(item["words"]),
                         key=lambda w: (w.bit_count(), w))[:6]
        translated = set()
        for seed in parents[:4]:
            positions, word = [], seed
            while word:
                bit = word & -word
                positions.append(divmod(bit.bit_length()-1, q))
                word ^= bit
            for permutation in translations[item["side"]]:
                moved = sum(1 << (j*q+permutation[h])
                            for j, h in positions)
                if moved and syndrome(item["cols"], moved) == 0:
                    translated.add(moved)
        discoveries = set(item["words"])
        ceiling = min((w.bit_count() for w in discoveries),
                      default=width+1)+4
        for word in parents:
            for other in translated:
                combined = word ^ other
                if combined and combined.bit_count() <= ceiling:
                    discoveries.add(combined)
        item["words"] = sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:32]
        item["d"] = min(item["d"], item["words"][0].bit_count())

    def refine(item, folded_trials, ordinary_trials, orbit=False):
        compact = item["compact"]
        nd = 2*probe(item["folded"], compact, folded_trials)
        expanded = [expand_central(w) for w in compact]
        assert all(syndrome(item["cols"], w) == 0 for w in expanded)
        words = sorted(set(item["words"]+expanded),
                       key=lambda w: (w.bit_count(), w))[:48]
        od = probe(item["cols"], words, ordinary_trials)
        item["words"] = words
        item["d"] = min(item["d"], nd, od)
        if orbit:
            orbit_refine(item)
        assert all(w and syndrome(item["cols"], w) == 0
                   for w in item["words"])

    def component_merit(item):
        d = min(item["d"], 36)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"])

    def component_pool(side):
        counts = ta if side == "A" else tb
        strata, seen = {}, set()

        def admit(grid):
            labels = tuple(tuple(sorted(cell))
                           for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            cols = compile_component(side, grid)
            rank = gf2_rank(cols)
            words = constant_fiber_words(cols)
            folded, compact = central_state(cols, words)
            item = dict(
                side=side, grid=grid, cols=cols, rank=rank,
                words=words, folded=folded, compact=compact,
                d=width+1, cycles=_four_cycles(component_rows(cols)))
            refine(item, 4, 2)
            bucket = strata.setdefault(rank, [])
            bucket.append(item)
            bucket.sort(key=component_merit, reverse=True)
            del bucket[3:]

        for attempt in range(96):
            admit(draw_grid(counts, (idx+attempt//32) % 3, side))

        for step in range(32):
            if not strata:
                admit(draw_grid(counts, step % 3, side))
                continue
            rank = rng.choice(sorted(strata))
            parent = rng.choice(strata[rank])
            grid = [[list(cell) for cell in row]
                    for row in parent["grid"]]
            fresh = draw_grid(counts, (idx+step) % 3, side)
            pressure = [
                1+sum(((w >> (j*q)) & mask).bit_count()
                      for w in parent["words"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = fresh[i][j]
            if step % 5 == 4:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = fresh[1-i][other]
            # Recompute witnesses after a voltage change; never inherit
            # a kernel word from the parent's different component.
            admit(grid)

        first = sorted((bucket[0] for bucket in strata.values()),
                       key=component_merit, reverse=True)
        pool = first[:8]
        extras = sorted(
            (item for bucket in strata.values() for item in bucket[1:]),
            key=component_merit, reverse=True)
        pool.extend(extras[:max(0, 8-len(pool))])
        for item in pool:
            refine(item, 20, 12, orbit=True)
        pool.sort(key=component_merit, reverse=True)
        pool = pool[:5]
        for item in pool:
            refine(item, 72, 48, orbit=True)
        return sorted(pool, key=component_merit, reverse=True)

    def full_reading(checks, stabilizers, n, seeds, trials):
        """Found nonzero logical classes, including mixed-sector words."""
        span = {}
        for original in stabilizers:
            word = original
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
        logicals, memory = {}, {}
        best = n+1

        def observe(word, cls):
            nonlocal best
            if word and cls:
                best = min(best, word.bit_count())
                logicals[word] = cls

        for word in seeds:
            if word and syndrome(cols, word) == 0:
                observe(word, logical_class(word))

        order = list(range(n))
        for trial in range(trials):
            rng.shuffle(order)
            if trial % 3 != 2:
                # Alternate elimination priorities for the two sectors.
                order.sort(key=lambda c:
                           (c < 9*q) if trial % 3 == 0 else (c >= 9*q))
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
            directions = dict(memory)
            directions.update(logicals)
            for word in sorted(set(dependencies), key=int.bit_count)[:96]:
                cls = logical_class(word)
                directions[word] = cls
                observe(word, cls)
            if not logicals:
                for word in dependencies:
                    cls = logical_class(word)
                    if cls:
                        directions[word] = cls
                        observe(word, cls)
                        break

            pairs = sorted(directions.items(),
                           key=lambda x: (x[0].bit_count(), x[0]))[:64]
            candidates = dict(logicals)
            for a, (word, cls) in enumerate(pairs):
                for other, oc in pairs[:a]:
                    combined, cc = word ^ other, cls ^ oc
                    if combined and cc:
                        observe(combined, cc)
                        candidates[combined] = cc
            reducers = pairs + [(row, 0) for row in stabilizers]
            rng.shuffle(reducers)
            starts = sorted(candidates.items(),
                            key=lambda x: (x[0].bit_count(), x[0]))[:12]
            for word, cls in starts:
                for sweep in range(3):
                    changed = False
                    for other, oc in reducers:
                        combined, cc = word ^ other, cls ^ oc
                        if cc and combined.bit_count() < word.bit_count():
                            word, cls = combined, cc
                            changed = True
                    if not changed:
                        break
                assert syndrome(cols, word) == 0
                observe(word, cls)

            logicals = dict(sorted(
                logicals.items(),
                key=lambda x: (x[0].bit_count(), x[0]))[:24])
            trivial = [(w, c) for w, c in directions.items() if not c]
            memory = dict(sorted(
                trivial, key=lambda x: (x[0].bit_count(), x[0]))[:16])
            memory.update(logicals)
        assert logicals and best <= n
        return best

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
            assert columns(ar) == a["cols"]
            assert columns(br) == b["cols"]
            proposals.append(dict(
                A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                ac=a["cols"], bc=b["cols"],
                aw=list(a["words"]), bw=list(b["words"]),
                da=a["d"], db=b["d"],
                cycles=a["cycles"]+b["cycles"]))

    if not proposals:
        raise ValueError("no independently screened Dic11 pair in cell 0.100")
    finalists = sorted(proposals, key=merit, reverse=True)[:4]
    for item in finalists:
        item["da"] = min(item["da"], probe(item["ac"], item["aw"], 24))
        item["db"] = min(item["db"], probe(item["bc"], item["bw"], 24))
        item["logical"] = dressed_reading(item)
        aseeds = [
            sum(((word >> (j*q)) & mask) << ((3*j+fixed)*q)
                for j in range(3))
            for word in item["aw"][:12] for fixed in range(3)]
        bseeds = [
            word << (fixed*3*q)
            for word in item["bw"][:12] for fixed in range(3)]
        item["logical"] = min(
            item["logical"],
            full_reading(item["hx"], item["hz"], item["n"], aseeds, 6),
            full_reading(item["hz"], item["hx"], item["n"], bseeds, 6))

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 572 and q <= k and 8*k < n
    assert css_ok(hx, hz)
    actual_a = [[len(cell) for cell in row] for row in A]
    actual_b = [[len(cell) for cell in row] for row in B]
    assert actual_a == sa and actual_b == sb
    assert _grid_weights(actual_a, actual_b) <= 12
    assert _no_dead_lines(actual_a, actual_b)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert all(
        len(cell) == len(set(cell)) <= 5
        and all(G.mul(G.inv(g), h) != involution
                for g in cell for h in cell if g != h)
        for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell without further label changes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured current-parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Normal-C11 indicators plus ordinary kernel rescreening and local voltage repairs eliminate the Dic11 weight-24 bottleneck; current-parent measured slots 0 and 7 retained d30s=24 at k=57 and 56, while slot 5 reached d30s=27 at k=44.
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
