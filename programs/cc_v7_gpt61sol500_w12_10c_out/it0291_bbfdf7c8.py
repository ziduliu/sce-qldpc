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
CONCEPT = ("C7 x C5, cyclic order 35, A=(2,3), B=(2,4): degree-four A columns and rank-stratified, C5/C7-fiber-screened B components should produce five fresh cell-0.150 codes exceeding frontier fitness 43.2364; measured evaluation can refute this.")
GROUP = ("L3 jump from cyclic C36 to genuinely different cyclic C35, encoded as C7 x C5. C35 is absent from BEDS; nearby C37 reached 43.2364, whereas the current C36 program scored 38.6716. Odd order makes F2[C35] semisimple, and C5/C7 factors supply additional constituent rank strata to investigate. This is an algebraic search opportunity, not a transferred distance claim. All matrix actions and full ranks use verified helpers.")
SHAPES = ("Keep (2,3)x(2,4) for cell 0.150: n=560, kb=70, guaranteed rate 1/8, admissible q<=37, and components with 105/70 and 140/70 variables/checks. Current-parent measured limiting families occupied sector-one Acol/Bcol blocks, frequently emphasizing Acol1; listed minima did not touch Arow/Brow. Exact complete lifted ranks retain every output in [0.125,0.175).")
FILLING = ("Twenty tasks cross five B incidence profiles with four intrinsic relative-order policies. A rows contain six terms and columns four; B rows contain eight and columns at most five, giving HX<=11 and HZ<=12. Compile actions only from verified unit-cell lifts. Retain syndrome-verified commuting cofactors and C5/C7-fiber kernel words, repair witness-heavy cells, preserve measured component rank strata, and independently deepen finalists. Exactly exclude complete logicals through four, search genuine logical classes and stabilizer dressings in both sectors, and reject equal uncoloured Tanner invariants. Found readings remain upper estimates, not distance certificates.")
FAMILY_NAME = "r015_C7xC5_23x24_rankstrata_fiber57_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    # C7 x C5 is cyclic C35; (1,1) has order 35.
    if not valid_q(q):
        raise ValueError("C7 x C5 has order 35")
    els = [(a, b) for b in range(5) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 5)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % 7, (b + d) % 5)

    def inv(g):
        a, b = red(g)
        return ((-a) % 7, (-b) % 5)

    def idx(g):
        a, b = red(g)
        return 7*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _support_designs():
    """Rectangular architectures distinguished by incidence degrees."""
    from itertools import product, permutations

    def profiles(ca, lower, upper):
        perms = tuple(permutations(range(ca)))
        result = []
        for t in product(range(4), repeat=2*ca):
            rows = (sum(t[:ca]), sum(t[ca:]))
            cols = tuple(t[j]+t[ca+j] for j in range(ca))
            if min(rows) < lower or max(rows) > upper:
                continue
            if min(cols) < 3 or max(cols) > 6:
                continue
            canonical = min(
                tuple(t[ca*r+c] for r in order for c in perm)
                for order in ((0, 1), (1, 0)) for perm in perms)
            if t != canonical:
                continue
            p0 = sum((t[j] & 1) << j for j in range(ca))
            p1 = sum((t[ca+j] & 1) << j for j in range(ca))
            rank = int(bool(p0 or p1))
            rank += int(bool(p0 and p1 and p0 != p1))
            result.append((t, rows, cols, rank))
        return result

    ap = profiles(3, 5, 8)
    bp = profiles(4, 6, 9)
    buckets = {}
    for ta, ar, ac, ka in ap:
        for tb, br, bc, kb in bp:
            if max(ar)+max(bc) > 12 or max(br)+max(ac) > 12:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc]
                + [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc]
                + [x+y for x in ar for y in br]))
            priority = (
                min(ac+bc), sum(ta)+sum(tb),
                min(ar+br), -max(ac+bc), -max(ar+br))
            buckets.setdefault((ka, kb), []).append(
                (priority, (checks, qubits), ta, tb))
    for bucket in buckets.values():
        bucket.sort(reverse=True)

    # Keep rank-deficient and full-augmentation-rank architectures.
    # Unequal degree signatures rule out a combined Tanner relabeling.
    strata = ((0, 0), (0, 1), (1, 0), (1, 1),
              (0, 2), (2, 0), (1, 2), (2, 1), (2, 2))
    positions = {s: 0 for s in strata}
    seen, designs = set(), []
    while True:
        added = False
        for stratum in strata:
            bucket = buckets.get(stratum, ())
            while positions[stratum] < len(bucket):
                _, fingerprint, ta, tb = bucket[positions[stratum]]
                positions[stratum] += 1
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                designs.append((ta, tb))
                added = True
                break
        if not added:
            break
    if len(designs) < 20:
        raise ValueError("fewer than twenty rectangular architectures")
    return tuple(designs)


# Historical C36 support enumeration is inactive on the new bed.
_SUPPORT_DESIGNS = ()


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
    """Extract the rectangular components from the VERIFIED LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    arows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i*cb)*q+r]
            packed = 0
            for j in range(ca):
                packed |= ((source >> (j*cb*q)) & mask) << (j*q)
            arows.append(packed)
    bmask = (1 << (cb*q))-1
    brows = [hz[a*q+r] & bmask for a in range(rb) for r in range(q)]
    return arows, brows


def _label_grid(rng, G, counts):
    """Primitive C36 doubletons; otherwise distinct C9 residues."""
    from math import gcd
    ca = len(counts)//2
    grid = []
    for i in range(2):
        row = []
        for j in range(ca):
            count = counts[ca*i+j]
            residues = rng.sample(range(9), count)
            phases = [rng.randrange(4) for _ in range(count)]
            if count == 2:
                residues[1] = rng.choice([
                    a for a in range(9)
                    if gcd((a-residues[0]) % 9, 9) == 1])
                phases[1] = (phases[0]+rng.choice((1, 3))) % 4
            row.append(list(zip(residues, phases)))
        grid.append(row)
    return grid


def _abelian_cofactor_seeds(G, grid, cols, inverse=False):
    """Three-column minor directions, verified on the actual component.

    For the four-column side, enumerate all four three-column subsets.
    These are group-algebra words, not a replacement matrix lift.
    """
    from itertools import combinations
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

    seeds = set()
    for selected in combinations(range(len(grid[0])), 3):
        minors = {}
        for omitted in selected:
            s, t = [j for j in selected if j != omitted]
            minors[omitted] = (
                product(entries[0][s], entries[1][t])
                ^ product(entries[0][t], entries[1][s]))
        for shift in ((0, 0), (1, 0), (0, 1), (2, 1)):
            word = 0
            for j, minor in minors.items():
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
                seeds.add(word)
    return sorted(seeds, key=lambda v: (v.bit_count(), v))


def _retired_c36_context(rng, G, idx):
    """Inactive historical C36 optimizer; never called on the C35 bed."""
    from collections import Counter
    from math import gcd

    assert G.q == 36
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:4]), list(tb[4:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q, width = G.q, 4*G.q

    # Projection onto C4 supplies actual normal-C9 fiber indicators.
    # Each side uses only its own number of protograph columns.
    pools = [[g for g in G.elements if g[1] == b] for b in range(4)]
    fibers = [sum(1 << G.idx(g) for g in pool) for pool in pools]
    fiber_words = [fiber << (j*q)
                   for j in range(4) for fiber in fibers]

    def draw_grid(counts, model):
        if model == 0:
            return _label_grid(rng, G, counts)
        ca = len(counts)//2
        grid = [[None]*ca for _ in range(2)]
        row_used = [Counter(), Counter()]
        cross_used = Counter()
        for i in range(2):
            for j in range(ca):
                count = counts[ca*i+j]
                chosen = None
                for attempt in range(6 if model == 1 else 12):
                    phase = rng.randrange(4)
                    if count == 2:
                        quota = [phase, (phase+rng.choice((1, 3))) % 4]
                    elif count == 3:
                        offsets = ((0, 1, 2), (0, 1, 1), (0, 1, 3))
                        quota = [(phase+t) % 4
                                 for t in offsets[(idx+i+j+model) % 3]]
                    else:
                        quota = [phase]*count
                    residues = rng.sample(range(9), count)
                    if count == 2:
                        residues[1] = rng.choice([
                            a for a in range(9)
                            if gcd((a-residues[0]) % 9, 9) == 1])
                    cell = list(zip(residues, quota))
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

    def columns(rows, count):
        out = [0]*count
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                lo = row & -row
                out[lo.bit_length()-1] |= bit
                row ^= lo
        return out

    def constant_fiber_words(cols):
        """Enumerate the exact kernel restricted to this side's C9 fibers."""
        small_cols = []
        for word in fiber_words[:(len(cols)//q)*4]:
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

    def central2_reading(cols, ordinary, trials):
        """Search the exact invariant space of z=(0,2).

        Its coordinates are disjoint pairs of physical coordinates.
        Restricted syndromes are XORs of verified component columns.
        Only an expanded nonzero kernel word supplies a reading.
        """
        half = q//2
        mask = (1 << half)-1
        blocks = len(cols)//q
        folded = [
            cols[j*q+a] ^ cols[j*q+half+a]
            for j in range(blocks) for a in range(half)]

        def syndrome(word):
            value = 0
            while word:
                bit = word & -word
                value ^= folded[bit.bit_length()-1]
                word ^= bit
            return value

        compact = set()
        for seed in ordinary:
            norm, compressed, invariant = 0, 0, True
            for j in range(blocks):
                low = (seed >> (j*q)) & mask
                high = (seed >> (j*q+half)) & mask
                norm |= (low ^ high) << (j*half)
                compressed |= low << (j*half)
                invariant &= low == high
            word = compressed if invariant else norm
            if word and syndrome(word) == 0:
                compact.add(word)
        compact = sorted(compact, key=lambda v: (v.bit_count(), v))
        probe(folded, compact, trials)

        expanded = []
        for word in compact:
            full = 0
            for j in range(blocks):
                block = (word >> (j*half)) & mask
                full |= (block | (block << half)) << (j*q)
            if full:
                expanded.append(full)
        ordinary[:] = sorted(
            set(ordinary+expanded), key=lambda v: (v.bit_count(), v))[:40]
        return min((word.bit_count() for word in expanded),
                   default=len(cols)+1)

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1), 28)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                min(item["da"], 32)+min(item["db"], 32),
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
                for fixed in range(4 if side == "A" else 3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((j*4+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*4*q)
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
        fingerprint = tuple(tuple(sorted(cell)) for grid in (A, B)
                            for row in grid for cell in row)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (125*n <= 1000*k < 175*n):
            return
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar, 3*q), columns(br, 4*q)
        aw, bw = constant_fiber_words(ac), constant_fiber_words(bc)
        aw.extend(_abelian_cofactor_seeds(G, A, ac))
        bw.extend(_abelian_cofactor_seeds(G, B, bc, inverse=True))
        da = min(central2_reading(ac, aw, 6), probe(ac, aw, 4))
        db = min(central2_reading(bc, bw, 6), probe(bc, bw, 4))
        item = {
            "A": A, "B": B, "k": k, "n": n, "hx": hx, "hz": hz,
            "ac": ac, "bc": bc, "aw": aw, "bw": bw,
            "da": da, "db": db,
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    for attempt in range(80):
        model = (idx + attempt//20) % 3
        admit(draw_grid(ta, model), draw_grid(tb, model))

    # The support fingerprint remains fixed for the slot. Repairs alter
    # actual cell voltages, preferentially where short witnesses live,
    # rather than applying a common automorphism or block permutation.
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
        count_columns = 3 if side == "A" else 4
        pressure = [
            1 + sum(((word >> (j*q)) & mask).bit_count()
                    for word in seeds)
            for j in range(count_columns)]
        j = rng.choices(range(count_columns), weights=pressure, k=1)[0]
        i = rng.randrange(2)
        counts = ta if side == "A" else tb
        replacement = draw_grid(counts, (idx+step//8) % 3)
        grid = A if side == "A" else B
        grid[i][j] = replacement[i][j]
        if step % 5 == 0:
            # Repairs preserve every cell size and hence both weight caps.
            other = (j+1+rng.randrange(count_columns-1)) % count_columns
            grid[1-i][other] = replacement[1-i][other]
        admit(A, B)

    if not finalists:
        raise ValueError("no positive-k labeling in target rate cell")
    for item in finalists:
        item["da"] = min(
            item["da"],
            central2_reading(item["ac"], item["aw"], 24),
            probe(item["ac"], item["aw"], 16))
        item["db"] = min(
            item["db"],
            central2_reading(item["bc"], item["bw"], 24),
            probe(item["bc"], item["bw"], 16))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(
            item["da"],
            central2_reading(item["ac"], item["aw"], 64),
            probe(item["ac"], item["aw"], 96))
        item["db"] = min(
            item["db"],
            central2_reading(item["bc"], item["bw"], 64),
            probe(item["bc"], item["bw"], 96))
        item["logical"] = dressed_reading(item)
    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert n == 576 and k >= 72
    assert 125*n <= 1000*k < 175*n
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert all(
        len(cell) == len(set(cell)) <= 5
        and len({g[0] for g in cell}) == len(cell)
        and (len(cell) != 2 or (
            gcd((cell[1][0]-cell[0][0]) % 9, 9) == 1
            and (cell[1][1]-cell[0][1]) % 2 == 1))
        for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


# B rows sum to eight; column maxima are 4,5,4,5,4.
# A rows sum to six and columns to four:
# HX <= 6+5=11, HZ <= 8+4=12.
_C35_B_PROFILES = (
    ((2,2,2,2), (1,1,3,3)),
    ((3,2,2,1), (1,2,1,4)),
    ((3,3,1,1), (1,1,3,3)),
    ((3,2,2,1), (1,1,2,4)),
    ((2,2,2,2), (2,2,2,2)),
)


def _c35_columns(rows, width):
    cols = [0]*width
    for r, original in enumerate(rows):
        word = original
        while word:
            bit = word & -word
            cols[bit.bit_length()-1] |= 1 << r
            word ^= bit
    return tuple(cols)


def _c35_rows(cols, height):
    rows = [0]*height
    for c, original in enumerate(cols):
        syndrome = original
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length()-1] |= 1 << c
            syndrome ^= bit
    return rows


def _c35_basis(rows):
    basis = {}
    for original in rows:
        word = original
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _c35_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c35_short(cols, stabilizers=()):
    """Exact kernel/logical existence through four, modulo the given span."""
    basis = _c35_basis(stabilizers)
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[i]
        elif _c35_outside(bits[i] ^ old, basis):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _c35_outside(word ^ old, basis):
                return True
    return False


def _c35_syndrome(word, cols):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= cols[bit.bit_length()-1]
        word ^= bit
    return syndrome


def _c35_words(cols, local, trials, seeds=()):
    """Retain true kernel supports; unsuccessful searches are not proofs."""
    memory = sorted(
        {w for w in seeds if w and _c35_syndrome(w, cols) == 0},
        key=lambda w: (w.bit_count(), w))[:40]
    order = list(range(len(cols)))
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
            if not syndrome:
                dependencies.append(word)

        words = sorted(set(memory+dependencies),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        discoveries.update(words[:40])
        for sweep in range(2):
            donors = words[:28]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
            discoveries.update(words[:40])
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        if trial % 4 == 0:
            for i, word in enumerate(short[:8]):
                for j in range(i):
                    pair = word ^ short[j]
                    for h in range(j):
                        candidate = pair ^ short[h]
                        if candidate:
                            discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:40]
        if memory and memory[0].bit_count() <= 4:
            break
    return tuple(memory)


def _c35_fiber_words(cols, G, local, trials):
    """Expanded actual kernels in C5/C7-invariant subspaces.

    Indicators construct candidate supports only. Their syndromes are
    sums of verified component columns; no quotient lift is implemented.
    """
    q = G.q
    found = set()
    for size in (5, 7):
        fibers = [
            sum(1 << G.idx(g) for g in G.elements
                if (g[0] if size == 5 else g[1]) == value)
            for value in range(7 if size == 5 else 5)]
        supports = [
            fiber << (j*q)
            for j in range(len(cols)//q) for fiber in fibers]
        projected = tuple(_c35_syndrome(w, cols) for w in supports)
        for dependency in _c35_words(projected, local, trials):
            expanded = 0
            while dependency:
                bit = dependency & -dependency
                expanded ^= supports[bit.bit_length()-1]
                dependency ^= bit
            if expanded:
                assert _c35_syndrome(expanded, cols) == 0
                found.add(expanded)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:40])


def _c35_draw(local, G, counts, mode, side):
    """Pack actual differences under four intrinsic order policies."""
    from collections import Counter
    from math import gcd

    q = G.q
    step_pools = {
        divisor: [v for v in range(1, q) if gcd(v, q) == divisor]
        for divisor in (1, 5, 7)}
    grid = [[None]*len(counts[0]) for _ in range(2)]
    row_used = [Counter(), Counter()]
    cross_used = Counter()
    for i, sizes in enumerate(counts):
        for j, size in enumerate(sizes):
            best = None
            for attempt in range(12):
                if size == 1:
                    cell = [G.pick(local)]
                elif size == 2:
                    # Top/bottom order trades are intrinsic cell policies,
                    # not a common group automorphism of one labeling.
                    divisor = 1
                    if mode == 1 and i == 1:
                        divisor = 5
                    elif mode == 2 and i == 1:
                        divisor = 7
                    elif mode == 3:
                        divisor = (7 if i == 1 and j % 2 == 0 else
                                   5 if i == 0 and j % 2 == 1 else 1)
                    step = local.choice(step_pools[divisor])
                    anchor = G.pick(local)
                    cell = [
                        anchor,
                        G.mul(anchor, (step % 7, step % 5))]
                else:
                    cell = local.sample(G.elements, size)
                ratios = [
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h]
                crossing = ([] if i == 0 else [
                    G.mul(g, G.inv(h))
                    for g in grid[0][j] for h in cell])
                penalty = (
                    len(ratios)-len(set(ratios))
                    + sum(row_used[i][g] for g in ratios)
                    + len(crossing)-len(set(crossing))
                    + sum(cross_used[g] for g in crossing))
                key = (penalty, local.random())
                if best is None or key < best[0]:
                    best = (key, cell, ratios, crossing)
            _, cell, ratios, crossing = best
            assert len(cell) == len(set(cell)) == size
            grid[i][j] = cell
            row_used[i].update(ratios)
            cross_used.update(crossing)
    return grid


def _c35_logical(checks, stabilizers, n, local, trials, seeds=()):
    """Search genuine logical classes, retaining them separately from checks."""
    cols = _c35_columns(checks, n)
    span = _c35_basis(stabilizers)
    memory = set()
    best = n+1

    def observe(word):
        nonlocal best
        if word and _c35_outside(word, span):
            best = min(best, word.bit_count())
            memory.add(word)

    for word in seeds:
        if word and _c35_syndrome(word, cols) == 0:
            observe(word)
    order = list(range(n))
    split = A_SHAPE[1]*B_SHAPE[1]*35
    for trial in range(trials):
        local.shuffle(order)
        if trial % 3 != 2:
            order.sort(key=lambda c:
                       c >= split if trial % 3 == 0 else c < split)
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

        logical = sorted(
            {w for w in dependencies if _c35_outside(w, span)} | memory,
            key=lambda w: (w.bit_count(), w))[:12]
        donors = sorted(dependencies, key=int.bit_count)[:40]
        discoveries = []
        for word in logical:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                for donor in donors:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if 0 < cw < weight and _c35_outside(candidate, span):
                        chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
            for sweep in range(2):
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda entry: entry[0])
                chosen, weight = word, word.bit_count()
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:10]]
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
            observe(word)
            discoveries.append(word)
        for i, word in enumerate(discoveries):
            for other in discoveries[:i]:
                observe(word ^ other)
        memory = set(sorted(memory,
                            key=lambda w: (w.bit_count(), w))[:24])
        if best <= 4:
            break
    retained = tuple(sorted(memory,
                            key=lambda w: (w.bit_count(), w)))
    assert all(_c35_syndrome(w, cols) == 0 for w in retained)
    return best, retained


def _c35_signature(hx, hz, n):
    """Uncoloured Tanner invariant; unequal signatures prove nonisomorphism.

    Equality is conservatively rejected, not asserted to prove isomorphism.
    Neither group labels nor X/Z colouring enters this signature.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0]*len(rows)
    neighborhoods = [Counter() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                neighborhoods[i][weights[j], overlap] += 1
                neighborhoods[j][weights[i], overlap] += 1
    profiles = []
    for i, neighbors in enumerate(adjacency):
        triangles = 0
        remaining = neighbors
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        profiles.append((
            weights[i], tuple(sorted(neighborhoods[i].items())),
            triangles//2))
    degrees = Counter(c.bit_count() for c in _c35_columns(rows, n))
    return (n, tuple(sorted(degrees.items())),
            tuple(sorted(Counter(profiles).items())))


@__import__("functools").lru_cache(maxsize=1)
def _c35_bank(G):
    """Pure deterministic bank of independently optimized structural tasks."""
    assert G.q == 35
    assert A_SHAPE == (2,3) and B_SHAPE == (2,4)
    q = G.q
    mask = (1 << q)-1
    za = [[[] for _ in range(3)] for _ in range(2)]
    zb = [[[] for _ in range(4)] for _ in range(2)]

    # Each atlas entry is extracted from a VERIFIED unit-cell LP lift.
    # Subsequent compilation is F2-linear superposition of those entries.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(3 if side == "A" else 4)]
                    for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zb) if side == "A" else (za, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            cols = _c35_columns(
                ar if side == "A" else br,
                (3 if side == "A" else 4)*q)
            blocks.append(cols[:q])
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*(len(grid[0])*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, syndrome in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+t] ^= syndrome << (i*q)
        return tuple(cols)

    def component_key(item):
        d = min(item["d"], 26)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"])

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("dx", item["n"]+1),
                item.get("dz", item["n"]+1), 26)
        return (item["k"]*d*d/item["n"], d,
                -item["a"]["cycles"]-item["b"]["cycles"])

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        lane = slot//20
        sa = ((2,2,2), (2,2,2))
        sb = _C35_B_PROFILES[(architecture+lane) % 5]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        local = random.Random(
            BASE_SEED*100003 + 35015791 + 104729*slot)

        def component_pool(side, batch):
            counts = sa if side == "A" else sb
            strata, seen = {}, set()

            def admit(grid):
                labels = tuple(tuple(sorted(cell))
                               for row in grid for cell in row)
                if labels in seen:
                    return
                seen.add(labels)
                cols = compile_component(side, grid)
                if _c35_short(cols):
                    return
                rows = _c35_rows(cols, 2*q)
                seeds = _abelian_cofactor_seeds(
                    G, grid, cols, inverse=(side == "B"))
                seeds.extend(_c35_fiber_words(cols, G, local, 2))
                words = _c35_words(cols, local, 3, seeds)
                if not words:
                    return
                item = dict(
                    grid=grid, cols=cols, words=words,
                    d=words[0].bit_count(), rank=gf2_rank(rows),
                    cycles=_four_cycles(rows))
                bucket = strata.setdefault(item["rank"], [])
                bucket.append(item)
                bucket.sort(key=component_key, reverse=True)
                del bucket[3:]

            quota = ((64 if side == "A" else 112)
                     + (24 if side == "A" else 40)*batch)
            for attempt in range(quota):
                admit(_c35_draw(local, G, counts, mode, side))

            positions = [(i,j) for i,row in enumerate(counts)
                         for j,count in enumerate(row) if count]
            for step in range(24 if side == "A" else 48):
                if not strata:
                    admit(_c35_draw(local, G, counts, mode, side))
                    continue
                rank = local.choice(sorted(strata))
                parent = local.choice(strata[rank])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                fresh = _c35_draw(local, G, counts, mode, side)
                pressure = [
                    1+sum(((w >> (j*q)) & mask).bit_count()
                          for w in parent["words"][:8])
                    for i,j in positions]
                for move in range(2 if step % 6 == 5 else 1):
                    i,j = local.choices(
                        positions, weights=pressure, k=1)[0]
                    grid[i][j] = fresh[i][j]
                # Changed matrices receive newly verified witnesses.
                admit(grid)

            first = sorted((bucket[0] for bucket in strata.values()),
                           key=component_key, reverse=True)[:8]
            extras = sorted(
                (item for bucket in strata.values() for item in bucket[1:]),
                key=component_key, reverse=True)
            pool = first+extras[:max(0,8-len(first))]
            for item in pool:
                seeds = tuple(item["words"])+_c35_fiber_words(
                    item["cols"], G, local, 12)
                item["words"] = _c35_words(
                    item["cols"], local, 24, seeds)
                item["d"] = min(item["d"],
                                item["words"][0].bit_count())
            pool.sort(key=component_key, reverse=True)

            # Preserve competitive measured rank strata, not presumed LP k.
            selected, ranks = [], set()
            for item in pool:
                if item["rank"] not in ranks:
                    selected.append(item)
                    ranks.add(item["rank"])
                if len(selected) == 4:
                    break
            for item in pool:
                if len(selected) == 4:
                    break
                if all(item is not old for old in selected):
                    selected.append(item)
            for item in selected:
                item["words"] = _c35_words(
                    item["cols"], local, 48, item["words"])
                item["d"] = min(item["d"],
                                item["words"][0].bit_count())
            return sorted(selected, key=component_key, reverse=True)

        chosen = None
        for batch in range(3):
            ap = component_pool("A", batch)
            bp = component_pool("B", batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (125*n <= 1000*k < 175*n):
                        continue
                    ar, br = _component_checks(hx, hz, q)
                    assert _c35_columns(ar, 3*q) == a["cols"]
                    assert _c35_columns(br, 4*q) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            proposals.sort(key=merit, reverse=True)

            finalists = []
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fp = _c35_signature(hx, hz, n)
                if fp in emitted:
                    continue
                if _c35_short(_c35_columns(hx, n), hz):
                    continue
                if _c35_short(_c35_columns(hz, n), hx):
                    continue
                az = tuple(
                    sum(((word >> (j*q)) & mask)
                        << ((j*4+beta)*q) for j in range(3))
                    for word in item["a"]["words"] for beta in range(4))
                bx = tuple(
                    word << (j*4*q)
                    for word in item["b"]["words"] for j in range(3))
                assert all(
                    not any((w & row).bit_count() & 1 for row in hx)
                    for w in az)
                assert all(
                    not any((w & row).bit_count() & 1 for row in hz)
                    for w in bx)
                dx, wx = _c35_logical(hz, hx, n, local, 3, bx)
                dz, wz = _c35_logical(hx, hz, n, local, 3, az)
                if dx > n or dz > n:
                    continue
                item.update(
                    dx=dx, dz=dz, wx=wx, wz=wz,
                    az=az, bx=bx, fingerprint=fp)
                finalists.append(item)
                if len(finalists) == 3:
                    break

            # Every finalist receives the same independent deeper budget.
            for item in finalists:
                dx, wx = _c35_logical(
                    item["hz"], item["hx"], item["n"], local, 8,
                    item["bx"]+item["wx"])
                dz, wz = _c35_logical(
                    item["hx"], item["hz"], item["n"], local, 8,
                    item["az"]+item["wz"])
                item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
            if not finalists:
                continue
            winner = max(finalists, key=merit)
            assert css_ok(winner["hx"], winner["hz"])
            assert winner["n"] == 560 and winner["k"] >= 70
            assert 125*560 <= 1000*winner["k"] < 175*560
            assert max(row.bit_count()
                       for row in winner["hx"]+winner["hz"]) <= W_MAX
            assert all(
                len(cell) == len(set(cell)) <= 5
                for grid in (winner["A"], winner["B"])
                for row in grid for cell in row)
            emitted.add(winner["fingerprint"])
            chosen = dict(
                cells={
                    side: tuple(tuple(tuple(cell) for cell in row)
                                for row in winner[side])
                    for side in ("A", "B")},
                k=winner["k"], card=(architecture, mode, lane),
                selection=merit(winner))
            break
        # Failed tasks never substitute an archived or transported code.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this card's fresh odd-order, rank-gated filling."""
    selected = _c35_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh C35 rank-stratified filling in cell 0.150")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further draws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive-order-46 doubletons eliminate the C46 sector-one weight-24 bottleneck; measured current-parent slots 0 and 6 retained weight-24 Z logicals at k=62, while slots 4,5,11 reached weight 28 at k=47,46,46.
# REFUTED: Primitive-order-36 doubletons and central-C2/component rescreening exceed the cell-0.150 record 41.1; the measured C36 program scored 38.6716, with four d30s=16 leaders and one d30s=17 leader, all reported minimum families confined to sector one.
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
