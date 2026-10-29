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
CONCEPT = ("C29 x C1, order 29, shapes (2,4)x(2,4): augmentation-zero doubletons with exact component rank 56, divided-cofactor witnesses and independent logical rescreening should produce five fresh [580,132] codes exceeding the cell-0.250 record of 43.5.")
GROUP = ("Keep the parent's two-cyclic-factor direct-product construction rule; only factor parameters and order change. Order 288 is inadmissible for the new factor twenty, whereas order 29 maximizes the lift and gives n=580. This bed is absent from BEDS. Odd order permits semisimple augmentation screening; the parent's measured bicycle fitness 53.2612 establishes neither superiority nor a distance guarantee for this bed.")
SHAPES = ("Target cell 0.250 using equal (2,4) sides: n=580, guaranteed kb=116 and rate 0.200, with equal 116-variable/58-check components. All-doubleton augmentation deficiencies supply additional dimensions; exact component rank 56 and assembled k=132 gates enforce rate 0.227586 inside the target cell. Parent minima occupied Acol0/Bcol0, Arow0/Brow0 or both, so genuine full-code probes inspect both sectors.")
FILLING = ("Twenty deliberate cards cross five relative-gap architectures with four bottom-row offset models. Every cell has two distinct terms; component rows total eight and columns four, giving HX=HZ=12. Compile only scaffold-generated unit actions, seed ordinary and common-binomial-divided commuting cofactors, retain true kernel words and recombine verified translations. Independently deepen component finalists, classify embedded witnesses modulo exact stabilizer spans, and search full-matrix logicals with single/pair check dressings. Require unequal connected uncoloured Tanner invariants between emitted codes. Sampled readings remain upper witnesses, never distance certificates.")
FAMILY_NAME = "r025_directproduct_C29xC1_24x24_augmentation_dividedcofactor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Same two-factor direct-product rule, now C29 x C1.
    if not valid_q(q):
        raise ValueError("C29 x C1 has order 29")
    m, r = 29, 1
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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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
    """Extract arbitrary-shaped components from the VERIFIED full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    arows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i*cb)*q+r]
            arows.append(sum(
                ((source >> (j*cb*q)) & mask) << (j*q)
                for j in range(ca)))
    bmask = (1 << (cb*q))-1
    brows = [hz[a*q+r] & bmask for a in range(rb) for r in range(q)]
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


def _unused_rectangular_ctx(rng, G, idx):
    """Historical rectangular optimizer; not called by the bicycle policy."""
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q, width = G.q, 3*G.q

    # Projection onto C2 defines six normal-C23 fiber indicators.
    # Their actual syndromes are obtained from the verified component.
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    fibers = [sum(1 << G.idx(g) for g in pool) for pool in pools]
    fiber_words = [fiber << (j*q)
                   for j in range(3) for fiber in fibers]

    def draw_grid(counts, model):
        if model == 0:
            return _label_grid(rng, G, counts)
        grid = []
        for i in range(2):
            row = []
            for j in range(3):
                count = counts[3*i+j]
                phase = rng.randrange(2)
                if count == 2:
                    quota = [phase, 1-phase]
                elif model == 1:
                    quota = [(phase+t) % 2 for t in range(count)]
                elif count == 1:
                    quota = [phase]
                elif count == 3:
                    quota = [phase, phase, 1-phase]
                else:
                    # Clustered 3+1 and balanced 2+2 occupancies are
                    # different cell models, not common group relabelings.
                    quota = ([phase]*3 + [1-phase] if (idx+i+j) % 2
                             else [phase]*2 + [1-phase]*2)
                rng.shuffle(quota)
                residues = rng.sample(range(23), count)
                row.append(list(zip(residues, quota)))
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

    finalists, seen = [], set()

    def admit(A, B):
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
        da, db = probe(ac, aw, 3), probe(bc, bw, 3)
        item = {
            "A": A, "B": B, "k": k, "n": n, "hx": hx, "hz": hz,
            "ac": ac, "bc": bc, "aw": aw, "bw": bw,
            "da": da, "db": db,
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    for attempt in range(64):
        model = (idx + attempt//16) % 3
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
        item["da"] = min(item["da"],
                         probe(item["ac"], item["aw"], 16))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 16))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(item["da"],
                         probe(item["ac"], item["aw"], 96))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 96))
        item["logical"] = dressed_reading(item)
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
        and len({g[0] for g in cell}) == len(cell)
        and (len(cell) != 2 or cell[0][1] != cell[1][1])
        for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


from functools import lru_cache


@lru_cache(maxsize=None)
def _bb_unit(G, side, element):
    """Full unit-cell matrices from the VERIFIED scaffold, not another lift."""
    A, B = [[[]]], [[[]]]
    (A if side == "A" else B)[0][0] = [G.elements[element]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert n == 2*G.q
    return tuple(hx), tuple(hz)


def _bb_build(G, a, b):
    """F2-linear superposition of verified unit-cell matrices."""
    hx, hz = [0]*G.q, [0]*G.q
    for side, cell in (("A", a), ("B", b)):
        for g in cell:
            ux, uz = _bb_unit(G, side, G.idx(g))
            for r in range(G.q):
                hx[r] ^= ux[r]
                hz[r] ^= uz[r]
    return hx, hz, 2*G.q


def _bb_columns(rows, n):
    cols = [0]*n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _bb_basis(rows):
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


def _bb_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _bb_dependencies(rng, cols, atoms):
    order = list(range(len(cols)))
    rng.shuffle(order)
    pivots, words = {}, []
    for c in order:
        syndrome, word = cols[c], atoms[c]
        while syndrome:
            p = syndrome.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = (syndrome, word)
                break
            syndrome ^= old[0]
            word ^= old[1]
        if not syndrome:
            words.append(word)
    return words


def _bb_search(rng, G, checks, stabilizers, trials, seeds=(), folded=False):
    """Actual full-code logical witnesses; no sentinel earns a high score."""
    n, q = 2*G.q, G.q
    cols = _bb_columns(checks, n)
    span = _bb_basis(stabilizers)
    atoms = [1 << c for c in range(n)]
    logicals, memory = set(), []
    best = n+1

    def observe(word):
        nonlocal best
        if word and _bb_outside(word, span):
            logicals.add(word)
            best = min(best, word.bit_count())

    for word in seeds:
        if not any((word & row).bit_count() & 1 for row in checks):
            observe(word)

    restrictions = []
    if folded:
        for involution in ((12, 0), (0, 6), (12, 6)):
            small_cols, small_atoms = [], []
            for sector in range(2):
                unseen = set(range(q))
                while unseen:
                    a = min(unseen)
                    b = G.idx(G.mul(G.elements[a], involution))
                    unseen.difference_update((a, b))
                    ca, cb = sector*q+a, sector*q+b
                    small_cols.append(cols[ca] ^ cols[cb])
                    small_atoms.append((1 << ca) | (1 << cb))
            restrictions.append((small_cols, small_atoms))

    for trial in range(trials):
        dependencies = _bb_dependencies(rng, cols, atoms)
        if restrictions:
            sc, sa = restrictions[trial % len(restrictions)]
            dependencies += _bb_dependencies(rng, sc, sa)
        for word in dependencies:
            observe(word)

        words = sorted(set(dependencies+memory),
                       key=lambda w: (w.bit_count(), w))[:48]
        discoveries = set(words)
        for i, word in enumerate(words):
            for other in words[:i]:
                combined = word ^ other
                if combined:
                    observe(combined)
                    discoveries.add(combined)

        # Keep logical seeds separately: short stabilizers must not
        # crowd every nontrivial class out of the retained witness memory.
        short_logicals = sorted(
            logicals, key=lambda w: (w.bit_count(), w))[:8]
        reducers = list(stabilizers) + words[:24]
        for word in short_logicals:
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = []
                for row in stabilizers:
                    cw = (word ^ row).bit_count()
                    nearby.append((cw, row))
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                nearby.sort(key=lambda entry: entry[0])
                moves = [row for _, row in nearby[:12]]
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                # Other kernel words may change the logical class.
                # Test that class exactly before accepting a shortening.
                for other in reducers:
                    candidate = word ^ other
                    if (candidate.bit_count() < weight
                            and _bb_outside(candidate, span)):
                        chosen, weight = candidate, candidate.bit_count()
                if chosen == word:
                    break
                word = chosen
                observe(word)
            discoveries.add(word)
        logicals = set(sorted(
            logicals, key=lambda w: (w.bit_count(), w))[:24])
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        if best <= 4:
            break
    return best, tuple(sorted(
        logicals, key=lambda w: (w.bit_count(), w))[:12])


def _bb_low4(checks, stabilizers, n):
    """Exact logical existence test through four, modulo stabilizers."""
    span = _bb_basis(stabilizers)
    cols = _bb_columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _bb_outside(word ^ old, span):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _bb_outside(word ^ old, span):
                return True
    return False


def _bb_signature(hx, hz, n):
    """Uncoloured Tanner invariants; unequal signatures certify novelty.

    Closed weighted walks in the check-intersection graph supplement
    degree and overlap profiles. Equal signatures are rejected
    conservatively; equality does not establish isomorphism.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    cols = _bb_columns(rows, n)

    # Connectivity of the full bipartite Tanner graph.
    unseen = set(range(len(rows)))
    reached, frontier = set(), [0]
    while frontier:
        r = frontier.pop()
        if r in reached:
            continue
        reached.add(r)
        word = rows[r]
        while word:
            bit = word & -word
            neighbors = cols[bit.bit_length()-1]
            word ^= bit
            while neighbors:
                low = neighbors & -neighbors
                s = low.bit_length()-1
                neighbors ^= low
                if s not in reached:
                    frontier.append(s)
    if len(reached) != len(rows):
        return None
    if any(not column for column in cols):
        return None

    adjacency = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap

    profiles = Counter()
    for i, neighbors in enumerate(adjacency):
        local = Counter((weights[j], w) for j, w in neighbors.items())
        second = Counter()
        for j, w in neighbors.items():
            for h, v in adjacency[j].items():
                second[h] += w*v
        third = sum(second[j]*w for j, w in neighbors.items())
        fourth = sum(value*value for value in second.values())
        profiles[(weights[i], tuple(sorted(local.items())),
                  third, fourth)] += 1
    return (tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(profiles.items())))


def _bb_draw(rng, G, count, model, side, reference=None):
    """Intrinsic support geometries, not automorphism copies of saved codes."""
    if model == 2 and reference is not None:
        # Preserve the odd-torus projection but independently alter
        # nilpotent coordinates. This is not a common translation.
        residues = [(g[0] % 3, g[1] % 3) for g in reference]
        cell = []
        for t in range(count):
            a, b = residues[t % len(residues)]
            legal = [(x, y) for x in range(a, 24, 3)
                     for y in range(b, 12, 3) if (x, y) not in cell]
            cell.append(rng.choice(legal))
        return cell

    if model in (0, 1, 2):
        split = 1 if count == 3 else count//2
        if side == "B":
            split = count-split
        for restart in range(64):
            if model == 1:
                directions = ((1, rng.randrange(12)),
                              (rng.randrange(24), 1))
            else:
                directions = ((1, 0), (0, 1))
            offsets = (rng.sample(range(1, 24), split)
                       + rng.sample(range(1, 12), count-split))
            cell = []
            for t, offset in enumerate(offsets):
                u, v = directions[int(t >= split)]
                cell.append((u*offset % 24, v*offset % 12))
            if len(set(cell)) == count:
                return cell

    winner = None
    for attempt in range(24):
        cell = rng.sample(G.elements, count)
        differences = [G.mul(g, G.inv(h))
                       for g in cell for h in cell if g != h]
        energy = len(differences)-len(set(differences))
        key = (energy, rng.random())
        if winner is None or key < winner[0]:
            winner = (key, cell)
        if energy == 0:
            break
    return winner[1]


@lru_cache(maxsize=1)
def _bb_bank(G):
    """Pure deterministic bank of independently optimized structural cards."""
    assert G.q == 288 and A_SHAPE == B_SHAPE == (1, 1)
    count_cards = ((3,3), (3,4), (3,5), (4,4), (5,5))
    emitted, bank = set(), []

    for slot in range(N_SLOTS):
        geometry, model = divmod(slot % 20, 4)
        na, nb = count_cards[geometry]
        local = random.Random(BASE_SEED*100003 + 28801091 + 104729*slot)
        strata, labelings = {}, set()

        for attempt in range(512 + 64*(slot//20)):
            a = _bb_draw(local, G, na, model, "A")
            b = _bb_draw(local, G, nb, model, "B", a)
            key = (tuple(sorted(a)), tuple(sorted(b)))
            if key in labelings:
                continue
            labelings.add(key)
            hx, hz, n = _bb_build(G, a, b)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (0 < k and 8*k < n):
                continue
            cycles = _four_cycles(hx)+_four_cycles(hz)
            item = dict(a=a, b=b, hx=hx, hz=hz, n=n, k=k,
                        cycles=cycles, attempt=attempt)
            bucket = strata.setdefault(k, [])
            bucket.append(item)
            bucket.sort(key=lambda x: (x["cycles"], x["attempt"]))
            del bucket[3:]

        raw = [item for bucket in strata.values() for item in bucket]
        # Preserve rank diversity before filling spare screening positions.
        first = sorted((bucket[0] for bucket in strata.values()),
                       key=lambda x: (x["k"], -x["cycles"]), reverse=True)
        pool = first[:24]
        extras = sorted(
            (item for bucket in strata.values() for item in bucket[1:]),
            key=lambda x: (x["k"], -x["cycles"]), reverse=True)
        pool.extend(extras[:max(0, 24-len(pool))])

        for item in pool:
            dz, wz = _bb_search(
                local, G, item["hx"], item["hz"], 3, folded=True)
            dx, wx = _bb_search(
                local, G, item["hz"], item["hx"], 3, folded=True)
            item.update(dx=dx, dz=dz, wx=wx, wz=wz)

        def merit(item):
            d = min(item["dx"], item["dz"], 40)
            return (item["k"]*d*d/item["n"], d, -item["cycles"])

        pool = [item for item in pool
                if max(item["dx"], item["dz"]) <= item["n"]]
        pool.sort(key=merit, reverse=True)
        finalists = pool[:8]
        # Equal fresh budgets for every finalist, retaining earlier minima.
        for item in finalists:
            dx, wx = _bb_search(
                local, G, item["hz"], item["hx"], 18,
                item["wx"], folded=True)
            dz, wz = _bb_search(
                local, G, item["hx"], item["hz"], 18,
                item["wz"], folded=True)
            item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
        finalists.sort(key=merit, reverse=True)

        selected = None
        for item in finalists:
            hx, hz, n = item["hx"], item["hz"], item["n"]
            if _bb_low4(hx, hz, n) or _bb_low4(hz, hx, n):
                continue
            signature = _bb_signature(hx, hz, n)
            if signature is None or signature in emitted:
                continue
            A, B = [[item["a"]]], [[item["b"]]]
            # Confirm the compiler against a fresh assembled verified lift.
            vx, vz, vn = lifted_matrices(G, A, B)
            assert hx == vx and hz == vz and n == vn == 576
            assert css_ok(vx, vz)
            assert lifted_k(G, A, B) == item["k"]
            assert 0 < item["k"] and 8*item["k"] < n
            assert max(row.bit_count() for row in vx+vz) <= 12
            assert _grid_weights([[na]], [[nb]]) <= 12
            assert _no_dead_lines([[na]], [[nb]])
            assert all(len(cell) == len(set(cell)) <= 5
                       for cell in (item["a"], item["b"]))
            emitted.add(signature)
            selected = dict(A=A, B=B, k=item["k"], card=(geometry, model),
                            selection=merit(item))
            break
        # Never substitute an emitted or archived code for a failed card.
        bank.append(selected)
    return tuple(bank)


def _lp29_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _lp29_draw(rng, kind, mode):
    """Gap architectures and intrinsic offset models, not saved-code orbits."""
    q = 29
    templates = (
        None, None,
        ((1,2,4,8), (3,5,7,11)),
        ((1,3,6,10), (2,4,8,13)),
        ((1,2,3,5), (4,6,9,14)),
    )
    if kind == 0:
        gaps = rng.sample(range(1, 15), 8)
        top, bottom = gaps[:4], gaps[4:]
    elif kind == 1:
        top = rng.sample(range(1, 15), 4)
        bottom = list(top)
        rng.shuffle(bottom)
    else:
        top, bottom = map(list, templates[kind])
    unit = rng.randrange(1, q)
    top = [unit*d % q for d in top]
    bottom = [unit*d % q for d in bottom]

    winner = None
    for _ in range(12):
        alpha, beta = rng.randrange(1, q), rng.randrange(1, q)
        phase = rng.randrange(q)
        if mode == 0:
            offsets = [rng.randrange(q) for _ in range(4)]
        elif mode == 1:
            offsets = [(phase+alpha*d+beta*e) % q
                       for d, e in zip(top, bottom)]
        elif mode == 2:
            offsets = [(phase+alpha*d*d+beta*e) % q
                       for d, e in zip(top, bottom)]
        else:
            offsets = rng.sample(range(1, q), 4)
        crossings = []
        for d, e, a in zip(top, bottom, offsets):
            crossings.extend((-a % q, (-a-e) % q,
                              (d-a) % q, (d-a-e) % q))
        energy = len(crossings)-len(set(crossings))
        key = (energy, rng.random())
        if winner is None or key < winner[0]:
            winner = (key, offsets)
    offsets = winner[1]
    return [
        [[(0,0), (d,0)] for d in top],
        [[(a,0), ((a+e) % q,0)]
         for a, e in zip(offsets, bottom)],
    ]


def _lp29_cofactors(G, grid, cols, inverse):
    """Ordinary and divided syzygies; every physical syndrome is checked.

    Polynomial division constructs candidate words, not a parity lift.
    Each even cell polynomial is divisible by 1+x before reduction
    modulo x^29+1. Minors of the divided entries are also tested.
    """
    from itertools import combinations
    q = G.q
    entries = [
        [set(G.idx(G.inv(g) if inverse else g) for g in cell)
         for cell in row] for row in grid]

    def divide(cell):
        polynomial = sum(1 << a for a in cell)
        quotient = 0
        while polynomial:
            p = polynomial.bit_length()-1
            assert p > 0, "even polynomial had nonzero remainder"
            quotient ^= 1 << (p-1)
            polynomial ^= 3 << (p-1)
        return {a for a in range(q) if quotient & (1 << a)}

    def product(left, right):
        answer = set()
        for a in left:
            for b in right:
                c = (a+b) % q
                if c in answer:
                    answer.remove(c)
                else:
                    answer.add(c)
        return answer

    divided = [[divide(cell) for cell in row] for row in entries]
    found = set()
    for matrix in (entries, divided):
        for triple in combinations(range(4), 3):
            word = 0
            for omitted in triple:
                s, t = [j for j in triple if j != omitted]
                minor = (product(matrix[0][s], matrix[1][t])
                         ^ product(matrix[0][t], matrix[1][s]))
                for a in minor:
                    word ^= 1 << (omitted*q+a)
            if word and _lp29_syndrome(cols, word) == 0:
                found.add(word)
    mask = (1 << q)-1
    for j in range(4):
        word = mask << (j*q)
        if _lp29_syndrome(cols, word) == 0:
            found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w)))


def _lp29_words(cols, rng, trials, seeds=(), orbit=False):
    """Bounded true-kernel search; retained readings never certify distance."""
    memory = sorted(
        {w for w in seeds if w and _lp29_syndrome(cols, w) == 0},
        key=lambda w: (w.bit_count(), w))[:32]
    atoms = [1 << c for c in range(len(cols))]
    for _ in range(trials):
        deps = _bb_dependencies(rng, cols, atoms)
        words = sorted(set(memory+deps),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        discoveries.update(words[:32])
        for sweep in range(2):
            donors = words[:24]
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
            discoveries.update(words[:32])
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]

    if orbit and memory:
        q, mask = 29, (1 << 29)-1

        def shift(word, t):
            return sum(
                ((((word >> (j*q)) & mask) << t
                  | (((word >> (j*q)) & mask) >> (q-t))) & mask)
                << (j*q) for j in range(4))

        discoveries = set(memory)
        short = memory[:8]
        for i, word in enumerate(short):
            for other in short[:i+1]:
                for t in range(q):
                    candidate = word ^ shift(other, t)
                    if (candidate
                            and _lp29_syndrome(cols, candidate) == 0):
                        discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
    assert memory
    assert all(_lp29_syndrome(cols, w) == 0 for w in memory)
    return tuple(memory)


def _lp29_logicals(checks, stabilizers, n, rng, trials, seeds):
    """Actual non-stabilizer witnesses, including both LP sectors."""
    cols = _bb_columns(checks, n)
    span = _bb_basis(stabilizers)
    found = set()
    best = n+1

    def observe(word):
        nonlocal best
        if word and _bb_outside(word, span):
            found.add(word)
            best = min(best, word.bit_count())

    for word in seeds:
        if word and _lp29_syndrome(cols, word) == 0:
            observe(word)
    atoms = [1 << c for c in range(n)]
    memory = []
    for trial in range(trials):
        deps = _bb_dependencies(rng, cols, atoms)
        for word in deps:
            observe(word)
        logicals = sorted(
            {w for w in deps if _bb_outside(w, span)} | found,
            key=lambda w: (w.bit_count(), w))[:12]
        reducers = sorted(set(deps+memory),
                          key=lambda w: (w.bit_count(), w))[:32]
        for start in logicals:
            word = start
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight and _bb_outside(candidate, span):
                        chosen, weight = candidate, cw
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda entry: entry[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:12]]
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
            observe(word)
        for i, word in enumerate(logicals):
            for other in logicals[:i]:
                observe(word ^ other)
        found = set(sorted(found,
                           key=lambda w: (w.bit_count(), w))[:24])
        memory = reducers
    return best, tuple(sorted(found,
                             key=lambda w: (w.bit_count(), w))[:12])


@lru_cache(maxsize=1)
def _lp29_bank(G):
    """Fresh deterministic rectangular cards with exact target-rate gates."""
    assert G.q == 29 and A_SHAPE == B_SHAPE == (2,4)
    q, width = G.q, 4*G.q
    mask = (1 << q)-1
    zero = [[[] for _ in range(4)] for _ in range(2)]

    # Compile ONLY F2 superpositions of scaffold-generated unit actions.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(4)] for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            rows = ar if side == "A" else br
            blocks.append(tuple(_bb_columns(rows, width)[:q]))
        atlas[side] = tuple(blocks)

    def component(side, grid):
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+t] ^= value << (i*q)
        rows = [0]*(2*q)
        for c, value in enumerate(cols):
            while value:
                bit = value & -value
                rows[bit.bit_length()-1] |= 1 << c
                value ^= bit
        return rows, tuple(cols)

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        local = random.Random(BASE_SEED*100003+29025093+104729*slot)

        def component_key(item):
            d = item["words"][0].bit_count()
            load = sum(w.bit_count() <= d+1 for w in item["words"])
            return (min(d, 24), -load, -item["cycles"])

        def pool(side, batch):
            kind = geometry if side == "A" else (geometry+2) % 5
            raw, seen = [], set()

            def admit(grid):
                labels = tuple(tuple(sorted(cell))
                               for row in grid for cell in row)
                if labels in seen:
                    return
                seen.add(labels)
                rows, cols = component(side, grid)
                if gf2_rank(rows) != 2*q-2:
                    return
                seeds = _lp29_cofactors(G, grid, cols, side == "B")
                words = _lp29_words(cols, local, 3, seeds)
                item = dict(grid=grid, rows=rows, cols=cols, words=words,
                            cycles=_four_cycles(rows))
                raw.append(item)
                raw.sort(key=component_key, reverse=True)
                del raw[8:]

            for attempt in range(72+32*batch+8*(slot//20)):
                admit(_lp29_draw(local, kind, mode))
            for step in range(24):
                if not raw:
                    admit(_lp29_draw(local, kind, mode))
                    continue
                parent = local.choice(raw[:4])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                pressure = [
                    1+sum(((w >> (j*q)) & mask).bit_count()
                          for w in parent["words"][:8])
                    for j in range(4)]
                j = local.choices(range(4), weights=pressure, k=1)[0]
                # Offset repairs preserve the gap architecture, but
                # alter an individual cell rather than a common gauge.
                a, b = grid[1][j]
                delta = local.randrange(1, q)
                grid[1][j] = [
                    ((a[0]+delta) % q,0), ((b[0]+delta) % q,0)]
                admit(grid)

            for item in raw:
                item["words"] = _lp29_words(
                    item["cols"], local, 16, item["words"], orbit=True)
            raw.sort(key=component_key, reverse=True)
            finalists = raw[:4]
            for item in finalists:
                item["words"] = _lp29_words(
                    item["cols"], local, 40, item["words"], orbit=True)
            return sorted(finalists, key=component_key, reverse=True)

        def merit(item):
            d = min(item["a"]["words"][0].bit_count(),
                    item["b"]["words"][0].bit_count(),
                    item.get("dx", 581), item.get("dz", 581), 24)
            return (item["k"]*d*d/580, d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        chosen = None
        for batch in range(3):
            ap, bp = pool("A", batch), pool("B", batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 580 or k != 132:
                        continue
                    ar, br = _component_checks(hx, hz, q)
                    assert tuple(_bb_columns(ar, width)) == a["cols"]
                    assert tuple(_bb_columns(br, width)) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            proposals.sort(key=merit, reverse=True)

            finalists = []
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                signature = _bb_signature(hx, hz, n)
                if signature is None or signature in emitted:
                    continue
                if _bb_low4(hx, hz, n) or _bb_low4(hz, hx, n):
                    continue
                aw = [
                    sum(((word >> (j*q)) & mask)
                        << ((j*4+beta)*q) for j in range(4))
                    for word in item["a"]["words"][:16]
                    for beta in range(4)]
                bw = [
                    word << (j*4*q)
                    for word in item["b"]["words"][:16]
                    for j in range(4)]
                assert all(_lp29_syndrome(_bb_columns(hx, n), w) == 0
                           for w in aw)
                assert all(_lp29_syndrome(_bb_columns(hz, n), w) == 0
                           for w in bw)
                dx, wx = _lp29_logicals(hz, hx, n, local, 3, bw)
                dz, wz = _lp29_logicals(hx, hz, n, local, 3, aw)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx, dz=dz, wx=wx, wz=wz,
                            aw=aw, bw=bw, signature=signature)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # Every joint finalist receives the same fresh deeper budget.
            for item in finalists:
                dx, wx = _lp29_logicals(
                    item["hz"], item["hx"], 580, local, 12,
                    tuple(item["bw"])+item["wx"])
                dz, wz = _lp29_logicals(
                    item["hx"], item["hz"], 580, local, 12,
                    tuple(item["aw"])+item["wz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
            if not finalists:
                continue
            winner = max(finalists, key=merit)
            A, B = winner["A"], winner["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert css_ok(hx, hz)
            assert lifted_k(G, A, B) == 132
            assert 225*n <= 1000*132 < 275*n
            assert _grid_weights([[2]*4]*2, [[2]*4]*2) == 12
            assert _no_dead_lines([[2]*4]*2, [[2]*4]*2)
            assert max(row.bit_count() for row in hx+hz) == 12
            assert all(len(cell) == len(set(cell)) == 2
                       for grid in (A, B) for row in grid for cell in row)
            emitted.add(winner["signature"])
            chosen = dict(
                A=A, B=B, k=132, card=(geometry, mode),
                selection=merit(winner))
            break
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's fresh augmentation-screened rectangular filling."""
    item = _lp29_bank(G)[idx]
    if item is None:
        raise ValueError("C29 rectangular card found no fresh target-rate filling")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return every complete cell of the selected (2,4)x(2,4) design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive order-46 doubletons eliminate the C46 sector-one weight-24 bottleneck; measured parent slots 0 and 6 retained weight-24 Z witnesses at k=62, whereas slots 4,5,11 reached weight 28 at k=47,46,46.
# REFUTED: C24 x C12 square-bicycle rank/folded screening exceeds fitness 60.3346 and supplies twenty distinct designs; the measured parent scored 53.2612, with numerous isomorphic duplicates and failed cards, and slot 59 fell from d2s=52 to d30s=44.
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
