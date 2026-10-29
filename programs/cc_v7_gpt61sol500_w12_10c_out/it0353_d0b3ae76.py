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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): independently selected row-seven constituents, central-C2-invariant witness screening and individual central-voltage repairs should exceed the parent's 51.9442 five-code fitness while retaining k/n<0.125.")
GROUP = ("Frozen L1 Dic11 presentation, encoding and order. RUN STATE records 54.9 on this bed versus 48.4 on the order-44 abelian bed. The central involution x^11 permits individual voltage changes preserving a restricted component matrix. All ordinary and restricted syndromes, assembled ranks and stabilizer spaces come from verified construction helpers.")
SHAPES = ("Frozen equal (2,3) sides for cell 0.100: n=572, kb=44, guaranteed rate 1/13, and equal length-132 components with 88 checks. The size window permits q<=46; Dic11 fixes q=44. Parent measured weight-24 minima occupied sector-one Acol/Bcol blocks, prominently Acol1/2 and Bcol1/2; none of its reported minima touched Arow/Brow. Exact assembled ranks enforce k/n<0.125.")
FILLING = ("Cross five canonical row-seven support pairs with four initial C4 occupancy policies; extra twenty-slot batches use further support pairs. Each component has column totals four/five, so both check maxima remain twelve. Pack relative differences and exclude involution-separated cell terms. Retain actual unrestricted and central-C2-invariant kernel witnesses, repair individual central voltages with verified folded-matrix preservation, independently deepen distinct component pools, then assemble and dress rank-legal pairs. Noncommutative cofactors are not assumed. Sampled readings remain witness-based heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r010_dic11_23x23_row7_centralfiber_voltage_v3"

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
    """Optimize row-seven components and their individual central voltages."""
    from itertools import product, permutations
    from collections import Counter

    assert G.q == 44
    q, width = G.q, 3*G.q
    permutations3 = tuple(permutations(range(3)))
    profiles = set()
    for margins in ((4, 5, 5), (5, 4, 5), (5, 5, 4)):
        for top in product(range(6), repeat=3):
            if sum(top) != 7:
                continue
            bottom = tuple(margins[j]-top[j] for j in range(3))
            if not all(0 <= count <= 5 for count in bottom):
                continue
            counts = tuple(top)+bottom
            canonical = min(
                tuple(counts[3*r+c] for r in order for c in perm)
                for order in ((0, 1), (1, 0))
                for perm in permutations3)
            profiles.add(canonical)

    ordered = sorted(profiles)
    pairs = [(a, b) for p, a in enumerate(ordered) for b in ordered[p:]]
    pairs.sort(key=lambda pair: (
        min(pair[0]+pair[1]),
        -(pair[0]+pair[1]).count(0),
        -sum(count*count for count in pair[0]+pair[1]),
        int(pair[0] != pair[1]), pair), reverse=True)
    if len(pairs) < 15:
        raise ValueError("insufficient canonical row-seven architectures")

    geometry, policy = divmod(idx % 20, 4)
    ta, tb = pairs[(5*(idx//20)+geometry) % len(pairs)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 7 for grid in (sa, sb) for row in grid)
    assert all(sorted(grid[0][j]+grid[1][j] for j in range(3))
               == [4, 5, 5] for grid in (sa, sb))

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

    def draw_grid(counts, model):
        """Fresh voltages under this card's initial quotient occupancy."""
        recipes = (
            {
                1: (0,), 2: (0, 1), 3: (0, 1, 2),
                4: (0, 1, 2, 3), 5: (0, 0, 1, 2, 3),
            },
            {
                1: (0,), 2: (0, 2), 3: (0, 0, 1),
                4: (0, 0, 1, 3), 5: (0, 0, 1, 1, 2),
            },
            {
                1: (0,), 2: (0, 1), 3: (0, 1, 3),
                4: (0, 1, 1, 2), 5: (0, 1, 1, 2, 3),
            },
            {
                1: (0,), 2: (0, 2), 3: (0, 0, 3),
                4: (0, 0, 2, 2), 5: (0, 0, 1, 2, 2),
            },
        )
        grid = [[None]*3 for _ in range(2)]
        internal = [Counter(), Counter()]
        cross_used = Counter()
        for i in range(2):
            for j in range(3):
                count = counts[3*i+j]
                if not count:
                    grid[i][j] = []
                    continue
                offsets = recipes[policy][count]
                if policy == 2 and count == 2 and (i+j) % 2:
                    offsets = (0, 2)

                chosen = None
                for attempt in range((1, 6, 12)[model]):
                    phase = rng.randrange(4)
                    direction = rng.choice((-1, 1))
                    quota = [(phase+direction*t) % 4 for t in offsets]
                    rng.shuffle(quota)
                    cell = []
                    for c in quota:
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
                        2*(len(ratios)-len(set(ratios)))
                        + sum(internal[i][g] for g in ratios)
                        + len(cross)-len(set(cross))
                        + sum(cross_used[g] for g in cross))
                    key = (penalty, rng.random())
                    if chosen is None or key < chosen[0]:
                        chosen = (key, cell, ratios, cross)
                _, cell, ratios, cross = chosen
                grid[i][j] = cell
                internal[i].update(ratios)
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

    def syndrome(cols, word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    def probe(cols, structured, trials):
        """Retain actual nonzero kernel words across fresh information sets."""
        length = len(cols)
        memory = sorted(
            {word for word in structured if word},
            key=lambda word: (word.bit_count(), word))[:32]
        best = min((word.bit_count() for word in memory), default=length+1)
        order = list(range(length))
        for trial in range(trials):
            rng.shuffle(order)
            pivots, dependencies = {}, []
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
                if not syn and word:
                    dependencies.append(word)
                    best = min(best, word.bit_count())

            words = sorted(set(memory+dependencies),
                           key=lambda word: (word.bit_count(), word))
            discoveries = set(words)
            for sweep in range(2):
                donors = words[:24]
                reduced = []
                changed = False
                for word in words:
                    weight = word.bit_count()
                    for donor in donors:
                        combined = word ^ donor
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                            changed = True
                    reduced.append(word)
                    best = min(best, weight)
                words = sorted(set(reduced),
                               key=lambda word: (word.bit_count(), word))
                discoveries.update(words)
                if not changed:
                    break

            short = sorted(discoveries,
                           key=lambda word: (word.bit_count(), word))[:32]
            for a, word in enumerate(short):
                for other in short[:a]:
                    combined = word ^ other
                    if combined:
                        best = min(best, combined.bit_count())
                        discoveries.add(combined)
            if trials >= 12 and trial % 8 == 0:
                small = short[:8]
                for a, word in enumerate(small):
                    for b in range(a):
                        for c in range(b):
                            combined = word ^ small[b] ^ small[c]
                            if combined:
                                best = min(best, combined.bit_count())
                                discoveries.add(combined)
            memory = sorted(
                discoveries, key=lambda word: (word.bit_count(), word))[:32]
            if best <= 4:
                break
        structured[:] = memory
        return best

    # Physical central-C2 indicators, not a separately implemented lift.
    # Each indicator has two disjoint coordinates, so physical weight is
    # exactly twice coefficient weight.
    pair_words = [
        (1 << (j*q+G.idx((a, b))))
        | (1 << (j*q+G.idx(((a+11) % 22, b))))
        for j in range(3) for b in range(2) for a in range(11)]
    small_mask = (1 << 11)-1

    def expand_norm(coefficient):
        word = 0
        while coefficient:
            bit = coefficient & -coefficient
            word ^= pair_words[bit.bit_length()-1]
            coefficient ^= bit
        return word

    def norm_state(cols, ordinary):
        """Exact restricted syndromes derived from verified component columns."""
        folded = [syndrome(cols, word) for word in pair_words]
        compact = set()
        for word in ordinary:
            norm, compressed, invariant = 0, 0, True
            for j in range(3):
                for b in range(2):
                    base = j*q+22*b
                    low = (word >> base) & small_mask
                    high = (word >> (base+11)) & small_mask
                    offset = 11*(2*j+b)
                    norm |= (low ^ high) << offset
                    compressed |= low << offset
                    invariant &= low == high
            # Multiplication by 1+x^11 gives an invariant word.
            # An already invariant word is compressed directly.
            coefficient = compressed if invariant else norm
            if coefficient and syndrome(folded, coefficient) == 0:
                compact.add(coefficient)
        return folded, sorted(
            compact, key=lambda word: (word.bit_count(), word))

    def refine_norm(folded, compact, ordinary, trials):
        reading = probe(folded, compact, trials)
        expanded = [expand_norm(word) for word in compact]
        ordinary[:] = sorted(
            set(ordinary+expanded),
            key=lambda word: (word.bit_count(), word))[:48]
        return 2*reading

    def row_basis(rows):
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

    def dressed_reading(item):
        """Only verified non-stabilizer logicals lower this quantum reading."""
        mask = (1 << q)-1
        best = item["n"]+1
        for side, seeds, checks, stabilizers in (
                ("A", item["aw"], item["hx"], item["hz"]),
                ("B", item["bw"], item["hz"], item["hx"])):
            span = row_basis(stabilizers)
            logicals = []
            for seed in sorted(set(seeds), key=int.bit_count)[:12]:
                for fixed in range(3):
                    word = (sum(
                        ((seed >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(3)) if side == "A"
                        else seed << (fixed*3*q))
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if outside(word, span):
                        logicals.append(word)
                        best = min(best, word.bit_count())

            for word in sorted(set(logicals), key=int.bit_count)[:8]:
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda entry: entry[0])
                    for cw, row in nearby:
                        if 0 < cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:12]]
                    for a, row in enumerate(moves):
                        for other in moves[:a]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
        return best

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1), 36)
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    def component_merit(item, side):
        dkey, words, cycles = (
            ("da", "aw", "acy") if side == "A" else ("db", "bw", "bcy"))
        d = min(item[dkey], 36)
        load = sum(word.bit_count() <= d+1 for word in item[words])
        return (d, -load, -item[cycles])

    finalists, seen = [], set()
    side_pools = {"A": [], "B": []}

    def labeling(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def admit(A, B, source=None, guard=None):
        key = (labeling(A), labeling(B))
        if key in seen:
            return
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (q <= k and 8*k < n):
            return
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar), columns(br)
        aw, bw = constant_fiber_words(ac), constant_fiber_words(bc)

        # A changed matrix may inherit only genuinely surviving words.
        if source is not None:
            aw.extend(word for word in source["aw"]
                      if word and syndrome(ac, word) == 0)
            bw.extend(word for word in source["bw"]
                      if word and syndrome(bc, word) == 0)
        an, af = norm_state(ac, aw)
        bn, bf = norm_state(bc, bw)
        if guard is not None:
            side, expected = guard
            assert (an if side == "A" else bn) == expected
        if source is not None:
            if an == source["an"]:
                af.extend(source["af"])
            if bn == source["bn"]:
                bf.extend(source["bf"])

        da = min(refine_norm(an, af, aw, 3), probe(ac, aw, 2))
        db = min(refine_norm(bn, bf, bw, 3), probe(bc, bw, 2))
        acy, bcy = _four_cycles(ar), _four_cycles(br)
        item = dict(
            A=A, B=B, hx=hx, hz=hz, n=n, k=k,
            ac=ac, bc=bc, aw=aw, bw=bw,
            an=an, bn=bn, af=af, bf=bf, da=da, db=db,
            acy=acy, bcy=bcy, cycles=acy+bcy,
            akey=key[0], bkey=key[1])
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

        # Keep distinct constituents independently. An exceptional A
        # need not be discarded merely because its sampled B was weak.
        for side, keyname in (("A", "akey"), ("B", "bkey")):
            pool = side_pools[side]
            if any(old[keyname] == item[keyname] for old in pool):
                continue
            pool.append(item)
            pool.sort(key=lambda entry: component_merit(entry, side),
                      reverse=True)
            del pool[6:]

    for attempt in range(96):
        model = (idx+attempt//32) % 3
        admit(draw_grid(ta, model), draw_grid(tb, model))

    mask = (1 << q)-1
    for step in range(24):
        if not finalists:
            admit(draw_grid(ta, step % 3), draw_grid(tb, step % 3))
            continue
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        side = (rng.choice(("A", "B")) if step % 5 == 0
                else "A" if parent["da"] <= parent["db"] else "B")

        if step % 4 == 3:
            # Changing quotient voltages can improve the limiting
            # invariant distance, which central-only repairs cannot.
            proposal = draw_grid(ta if side == "A" else tb,
                                 (idx+step) % 3)
            if side == "A":
                A = proposal
            else:
                B = proposal
            admit(A, B, source=parent)
            continue

        grid = A if side == "A" else B
        seeds = parent["aw"] if side == "A" else parent["bw"]
        pressure = [
            1+sum(((word >> (j*q)) & mask).bit_count()
                  for word in seeds[:8])
            for j in range(3)]
        legal_columns = [j for j in range(3)
                         if any(grid[i][j] for i in range(2))]
        j = rng.choices(
            legal_columns, weights=[pressure[j] for j in legal_columns],
            k=1)[0]
        legal_rows = [i for i in range(2) if grid[i][j]]
        i = rng.choice(legal_rows)
        cell = list(grid[i][j])
        positions = rng.sample(
            range(len(cell)), min(len(cell), 2 if step % 3 == 2 else 1))
        for position in positions:
            # Individual central changes, never a common slot relabeling.
            cell[position] = G.mul(involution, cell[position])
        assert len(cell) == len(set(cell))
        assert all(G.mul(G.inv(g), h) != involution
                   for p, g in enumerate(cell) for h in cell[:p])
        grid[i][j] = cell
        expected = parent["an"] if side == "A" else parent["bn"]
        admit(A, B, source=parent, guard=(side, expected))

    if not finalists:
        raise ValueError("no row-seven labeling in target rate cell")

    # Equal independent budgets for each distinct retained constituent.
    # Restricted witnesses are expanded before ordinary physical weighing.
    for side, dkey, ckey, wkey, nkey, fkey in (
            ("A", "da", "ac", "aw", "an", "af"),
            ("B", "db", "bc", "bw", "bn", "bf")):
        pool = side_pools[side]
        for item in pool:
            item[dkey] = min(
                item[dkey],
                refine_norm(item[nkey], item[fkey], item[wkey], 16),
                probe(item[ckey], item[wkey], 12))
        pool.sort(key=lambda entry: component_merit(entry, side),
                  reverse=True)
        del pool[4:]
        for item in pool:
            item[dkey] = min(
                item[dkey],
                refine_norm(item[nkey], item[fkey], item[wkey], 64),
                probe(item[ckey], item[wkey], 64))
            assert all(syndrome(item[ckey], word) == 0
                       for word in item[wkey])
        pool.sort(key=lambda entry: component_merit(entry, side),
                  reverse=True)

    # Component kernels depend only on their own side. Quantum ranks
    # do not: recombination is admitted only using the complete lift.
    assembled = []
    for a in side_pools["A"]:
        for b in side_pools["B"]:
            A, B = a["A"], b["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (q <= k and 8*k < n):
                continue
            assembled.append(dict(
                A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                ac=a["ac"], bc=b["bc"],
                aw=list(a["aw"]), bw=list(b["bw"]),
                da=a["da"], db=b["db"],
                cycles=a["acy"]+b["bcy"]))
    if not assembled:
        raise ValueError("independent constituents failed assembled rate gate")
    assembled.sort(key=merit, reverse=True)
    assembled = assembled[:4]
    for item in assembled:
        item["logical"] = dressed_reading(item)

    winner = max(assembled, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    ar, br = _component_checks(hx, hz, q)
    assert columns(ar) == winner["ac"]
    assert columns(br) == winner["bc"]
    assert css_ok(hx, hz)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 572 and q <= k and 8*k < n
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert all(sum(row) == 7 for grid in (sa, sb) for row in grid)
    assert all(
        len(cell) == len(set(cell)) <= 5
        and len({(g[0] % 11, g[1]) for g in cell}) == len(cell)
        for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently selected, central-voltage-repaired full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rank surplus alone predicts stronger readings on this bed; current-parent slot 0 had k=60 and d2s=22, while measured slots 11,22,32 had k=44 and d30s=26.
# REFUTED: Normal-C11-constant seeds and scalar component rescreening eliminate the sector-one weight-24 bottleneck; current-parent measured slots 2 and 23 retained d30s=24 at k=52 and 51.
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
