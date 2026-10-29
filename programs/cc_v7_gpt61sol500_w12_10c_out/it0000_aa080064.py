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
CONCEPT = ("C23 x C2, order 46, shapes (3,2)x(3,2): margin-transported supports, projected-difference packing and independently rescreened central-C2 witnesses should raise five-code fitness above 60.3346 while retaining every selected rate below 0.125.")
GROUP = ("Keep the exact C23 x C2 group, order and encoding. Its measured (2,3)x(2,3) bed leads cell 0.100 at 60.3346, versus 54.9 on the related order-44 nonabelian bed. The new shapes still allow order 46 and n=598. Central-C2 restricted syndromes and every ordinary component, rank and logical class come from verified lifts.")
SHAPES = ("Change both sides to (3,2) for cell 0.100: n=598, kb=46 and guaranteed rate 1/13. Both tall components have 92 variables and 138 checks; their transposed kernel maps have 138 variables and 92 checks. Parent slots 4,5,11 reached sector-one weight 28 across Acol/Bcol blocks, while slots 0,6 retained weight-24 Z witnesses; slot 0 also had mixed-sector X witnesses touching Arow0/Brow0/1. Simultaneous transposition is an LP duality, not itself a distance improvement. Fresh fillings and exact ranks enforce k/n<0.125.")
FILLING = ("Transport canonical degree-distinguished margins through zero and five-term cells, preserving both weight caps and nonzero lines. Cross balanced, clustered and mixed order-23/order-46 occupancy rules; pack projected C23 internal and cross-row differences. Extract sector-two kernel maps from verified tall-shape lifts, seed syndrome-tested cofactors and fiber words, and independently search ordinary and central-C2 invariant witnesses. Repair witness-heavy cells, deepen finalists equally, then search exact nonzero logical classes with kernel recombination and single/pair stabilizer reductions. Readings remain search heuristics and upper bounds, never distance certificates.")
FAMILY_NAME = "r010_C23xC2_32x32_margintransport_projected_norm_logical_v2"

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

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


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


def _retired_wide_ctx(rng, G, idx):
    """Inactive wide-shape reference; the tall-shape NEW_CTX below replaces it."""
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


def _t_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length()-1] |= bit
            row ^= low
    return cols


def _t_syndrome(cols, word):
    value = 0
    while word:
        low = word & -word
        value ^= cols[low.bit_length()-1]
        word ^= low
    return value


def _t_components(hx, hz, q):
    """Extract both sector-two kernel maps from the VERIFIED tall lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    n1 = ca*cb*q
    ar = []
    # Fix B-row zero; gather all A-row sector-two blocks.
    for j in range(ca):
        for r in range(q):
            source = hz[(j*rb)*q+r] >> n1
            ar.append(sum(
                ((source >> (i*rb*q)) & mask) << (i*q)
                for i in range(ra)))
    # Fix A-row zero; gather all B-row sector-two blocks.
    bm = (1 << (rb*q))-1
    br = [(hx[beta*q+r] >> n1) & bm
          for beta in range(cb) for r in range(q)]
    return ar, br


@__import__("functools").lru_cache(maxsize=None)
def _t_transports(counts):
    """Preserve wide margins, hence tall margins and Tanner degrees."""
    from itertools import product
    counts = tuple(counts)
    margins = tuple(counts[j]+counts[j+3] for j in range(3))
    total = sum(counts[:3])
    choices = []
    for top in product(range(6), repeat=3):
        bottom = tuple(margins[j]-top[j] for j in range(3))
        if sum(top) != total or not all(0 <= c <= 5 for c in bottom):
            continue
        minors = []
        for omitted in range(3):
            a, b = [j for j in range(3) if j != omitted]
            minors.append(top[a]*bottom[b]+top[b]*bottom[a])
        # Nominal cofactor capacity guides proposals, not distance claims.
        priority = (min(minors), sum(minors), -max(minors))
        choices.append((priority, tuple(top)+bottom))
    choices.sort(reverse=True)
    assert choices
    return tuple(counts for _, counts in choices[:16])


def _t_draw(rng, G, counts, policy, transport=True):
    """Fresh labels packed in the C23 projection, not common orbit images."""
    from collections import Counter
    if transport:
        counts = rng.choice(_t_transports(tuple(counts)))
    grid = [[None]*3 for _ in range(2)]
    internal = [Counter(), Counter()]
    cross_used = Counter()
    for i in range(2):
        for j in range(3):
            count = counts[3*i+j]
            if not count:
                grid[i][j] = []
                continue
            if count == 1:
                minority = 0
            elif count == 2:
                minority = (1 if policy == 0 else 0 if policy == 1
                            else int((i+j) % 3 != 0))
            elif policy == 0:
                minority = count//2
            elif policy == 1:
                minority = 1
            else:
                minority = count//2 if (i+j) % 2 else 1
            winner = None
            for attempt in range(16):
                phase = rng.randrange(2)
                parity = [phase]*(count-minority)+[1-phase]*minority
                rng.shuffle(parity)
                residues = rng.sample(range(23), count)
                cell = list(zip(residues, parity))
                differences = [
                    (g[0]-h[0]) % 23
                    for g in cell for h in cell if g != h]
                cross = ([] if i == 0 else [
                    (g[0]-h[0]) % 23
                    for g in grid[0][j] for h in cell])
                penalty = (
                    4*(len(differences)-len(set(differences)))
                    + 2*sum(internal[i][d] for d in differences)
                    + 4*(len(cross)-len(set(cross)))
                    + 3*sum(cross_used[d] for d in cross))
                key = (penalty, rng.random())
                if winner is None or key < winner[0]:
                    winner = (key, cell, differences, cross)
            _, cell, differences, cross = winner
            grid[i][j] = cell
            internal[i].update(differences)
            cross_used.update(cross)
    return grid


def _t_words(rng, cols, trials, seeds=()):
    """Actual nonzero kernel words; high readings never certify distance."""
    memory = sorted(set(seeds), key=lambda v: (v.bit_count(), v))[:32]
    assert all(v and _t_syndrome(cols, v) == 0 for v in memory)
    order = list(range(len(cols)))
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
        words = sorted(set(memory+dependencies),
                       key=lambda v: (v.bit_count(), v))
        discoveries = set(words)
        for sweep in range(2):
            reducers = sorted(set(words+memory),
                              key=lambda v: (v.bit_count(), v))[:40]
            rng.shuffle(reducers)
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda v: (v.bit_count(), v))
            discoveries.update(words)
        short = sorted(discoveries,
                       key=lambda v: (v.bit_count(), v))[:32]
        for a, word in enumerate(short):
            for other in short[:a]:
                combined = word ^ other
                if combined:
                    discoveries.add(combined)
        if trials >= 12 and trial % 4 == 0:
            for a, word in enumerate(short[:8]):
                for b in range(a):
                    for c in range(b):
                        combined = word ^ short[b] ^ short[c]
                        if combined:
                            discoveries.add(combined)
        memory = sorted(discoveries,
                        key=lambda v: (v.bit_count(), v))[:32]
        if memory and memory[0].bit_count() <= 4:
            break
    assert memory
    return list(memory)


def _t_fiber_seeds(G, cols):
    """All combinations of six C23 fiber indicators, with actual syndromes."""
    q = G.q
    fibers = [
        sum(1 << G.idx(g) for g in G.elements if g[1] == b)
        for b in range(2)]
    generators = [fiber << (j*q)
                  for j in range(3) for fiber in fibers]
    out = []
    for subset in range(1, 64):
        word = 0
        for j, generator in enumerate(generators):
            if subset & (1 << j):
                word ^= generator
        if _t_syndrome(cols, word) == 0:
            out.append(word)
    return out


def _t_fold(cols):
    """Restricted syndromes for central-C2 invariant supports."""
    return [cols[j*46+a] ^ cols[j*46+23+a]
            for j in range(3) for a in range(23)]


def _t_expand(word):
    mask = (1 << 23)-1
    full = 0
    for j in range(3):
        block = (word >> (23*j)) & mask
        full |= (block | (block << 23)) << (46*j)
    return full


def _t_refine(rng, item, ordinary_trials, folded_trials):
    for side in ("a", "b"):
        cols = item[side+"c"]
        folded = item[side+"f"]
        compact = _t_words(
            rng, folded, folded_trials, item[side+"compact"])
        expanded = [_t_expand(word) for word in compact]
        assert all(_t_syndrome(cols, word) == 0 for word in expanded)
        words = _t_words(
            rng, cols, ordinary_trials, item[side+"w"]+expanded)
        item[side+"compact"] = compact
        item[side+"w"] = words
        item["d"+side] = min(
            item["d"+side],
            words[0].bit_count(), 2*compact[0].bit_count())


def _t_logical_search(rng, checks, stabilizers, n, seeds, trials):
    """Found logical weights, classified modulo the EXACT stabilizer span.

    Every dependency uses verified full-code syndromes. Class-preserving
    descent and pair dressing can enter either LP sector.
    """
    pivots = {}
    for original in stabilizers:
        word = original
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    ordered = sorted(pivots.items(), reverse=True)

    def logical_class(word):
        for p, row in ordered:
            if (word >> p) & 1:
                word ^= row
        return word

    cols = _t_columns(checks, n)
    best = n+1
    logicals = {}
    memory = {}

    def observe(word, cls):
        nonlocal best
        if word and cls:
            best = min(best, word.bit_count())
            logicals[word] = cls

    for word in seeds:
        if word and _t_syndrome(cols, word) == 0:
            observe(word, logical_class(word))

    order = list(range(n))
    for trial in range(trials):
        rng.shuffle(order)
        elimination, dependencies = {}, []
        for c in order:
            syn, word = cols[c], 1 << c
            while syn:
                p = syn.bit_length()-1
                old = elimination.get(p)
                if old is None:
                    elimination[p] = (syn, word)
                    break
                syn ^= old[0]
                word ^= old[1]
            if not syn:
                dependencies.append(word)

        short = sorted(set(dependencies),
                       key=lambda v: (v.bit_count(), v))[:96]
        directions = dict(memory)
        directions.update(logicals)
        for word in short:
            cls = logical_class(word)
            directions[word] = cls
            observe(word, cls)
        # Positive k guarantees some dependency lies outside the span.
        if not logicals:
            for word in dependencies:
                cls = logical_class(word)
                if cls:
                    directions[word] = cls
                    observe(word, cls)
                    break

        pairs = sorted(directions.items(),
                       key=lambda pair: (pair[0].bit_count(), pair[0]))[:64]
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
                        key=lambda pair: (pair[0].bit_count(), pair[0]))[:16]
        for word, cls in starts:
            weight = word.bit_count()
            for sweep in range(3):
                changed = False
                for other, oc in reducers:
                    combined, cc = word ^ other, cls ^ oc
                    cw = combined.bit_count()
                    if cc and cw < weight:
                        word, cls, weight = combined, cc, cw
                        changed = True
                if not changed:
                    break
            observe(word, cls)

        # Neither one-check nor two-check reduction changes logical class.
        for word, cls in sorted(
                logicals.items(),
                key=lambda pair: (pair[0].bit_count(), pair[0]))[:4]:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda pair: pair[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
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
                observe(word, cls)

        logicals = dict(sorted(
            logicals.items(),
            key=lambda pair: (pair[0].bit_count(), pair[0]))[:24])
        trivial = [(word, cls) for word, cls in directions.items() if not cls]
        memory = dict(sorted(
            trivial, key=lambda pair: (pair[0].bit_count(), pair[0]))[:16])
        memory.update(logicals)
        if best <= 4:
            break
    assert logicals, "positive-k full search must find a logical class"
    return best


def NEW_CTX(rng, G, idx):
    """One distinct margin architecture, optimized on the actual tall bed."""
    assert G.q == 46
    assert A_SHAPE == (3, 2) and B_SHAPE == (3, 2)
    # New labels, not transpositions of saved or previously drawn codes.
    rng = random.Random(rng.getrandbits(64) ^ 0x23320346)
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    q, width = G.q, 3*G.q
    finalists, seen = [], set()

    def tall(grid):
        return [[list(grid[j][i]) for j in range(2)] for i in range(3)]

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("logical", item["n"]+1), 36)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d/item["n"], d, -load,
                min(item["da"], 40)+min(item["db"], 40),
                -item["cycles"])

    def admit(a, b):
        key = tuple(tuple(sorted(cell)) for grid in (a, b)
                    for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        A, B = tall(a), tall(b)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (q <= k and 8*k < n):
            return
        ar, br = _t_components(hx, hz, q)
        ac, bc = _t_columns(ar, width), _t_columns(br, width)
        aw = _t_fiber_seeds(G, ac)
        bw = _t_fiber_seeds(G, bc)
        # These orientations are accepted only after actual syndrome tests.
        aw.extend(_abelian_cofactor_seeds(G, a, ac, inverse=True))
        bw.extend(_abelian_cofactor_seeds(G, b, bc, inverse=False))
        af, bf = _t_fold(ac), _t_fold(bc)
        ca, cb = _t_words(rng, af, 6), _t_words(rng, bf, 6)
        aw = _t_words(rng, ac, 3, aw+[_t_expand(v) for v in ca])
        bw = _t_words(rng, bc, 3, bw+[_t_expand(v) for v in cb])
        item = dict(
            a=a, b=b, A=A, B=B, hx=hx, hz=hz, n=n, k=k,
            ac=ac, bc=bc, af=af, bf=bf,
            acompact=ca, bcompact=cb, aw=aw, bw=bw,
            da=min(aw[0].bit_count(), 2*ca[0].bit_count()),
            db=min(bw[0].bit_count(), 2*cb[0].bit_count()),
            cycles=_four_cycles(ar)+_four_cycles(br))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    # Distinct slot margins have different combined Tanner degree
    # signatures. Transport changes support geometry without losing them.
    for attempt in range(72):
        policy = (idx+attempt//24) % 3
        admit(_t_draw(rng, G, ta, policy),
              _t_draw(rng, G, tb, (policy+idx//3) % 3))

    mask = (1 << q)-1
    for step in range(32):
        if not finalists:
            admit(_t_draw(rng, G, ta, step % 3),
                  _t_draw(rng, G, tb, (step+1) % 3))
            continue
        parent = rng.choice(finalists[:min(4, len(finalists))])
        a = [[list(cell) for cell in row] for row in parent["a"]]
        b = [[list(cell) for cell in row] for row in parent["b"]]
        side = ("a" if parent["da"] <= parent["db"] else "b")
        if step % 5 == 0:
            side = rng.choice(("a", "b"))
        grid = a if side == "a" else b
        counts = tuple(len(cell) for row in grid for cell in row)
        policy = (idx+step) % 3
        if step % 8 == 7:
            proposal = _t_draw(rng, G, counts, policy, transport=True)
            if side == "a":
                a = proposal
            else:
                b = proposal
        else:
            proposal = _t_draw(rng, G, counts, policy, transport=False)
            pressure = [
                1+sum(((word >> (j*q)) & mask).bit_count()
                      for word in parent[side+"w"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = proposal[i][j]
            if step % 6 == 5:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = proposal[1-i][other]
        admit(a, b)

    if not finalists:
        raise ValueError("no fresh tall-shape filling in cell 0.100")
    for item in finalists:
        _t_refine(rng, item, ordinary_trials=12, folded_trials=24)
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        _t_refine(rng, item, ordinary_trials=48, folded_trials=96)
    finalists.sort(key=merit, reverse=True)

    # Actual sector-two embeddings; every seed's full syndrome is tested
    # again by the stabilizer-classified search.
    n1 = 4*q
    for item in finalists[:3]:
        aseeds = [
            sum(((word >> (i*q)) & mask)
                << (n1+(3*i+fixed)*q) for i in range(3))
            for word in item["aw"][:16] for fixed in range(3)]
        bseeds = [
            word << (n1+fixed*3*q)
            for word in item["bw"][:16] for fixed in range(3)]
        dx = _t_logical_search(
            rng, item["hz"], item["hx"], item["n"], aseeds, 6)
        dz = _t_logical_search(
            rng, item["hx"], item["hz"], item["n"], bseeds, 6)
        item["logical"] = min(dx, dz)
    winner = max(finalists[:3], key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 598 and q <= k and 8*k < n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= 12 and _no_dead_lines(sa, sb)
    assert all(
        len(cell) == len(set(cell)) <= 5
        and len({g[0] for g in cell}) == len(cell)
        for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": k, "screen": merit(winner)}


def CELL(rng, G, side, i, j, ctx):
    """Return every cell of the selected tall-shape filling without redraw."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: Dic11 minimum logicals are always sector-one-only; measured parent slot 0 contained two weight-24 X logicals touching both sectors and Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Primitive order-46 doubletons eliminate the C46 sector-one weight-24 bottleneck; measured current-parent slots 0 and 6 retained weight-24 Z logicals at k=62, while slots 4,5,11 reached weight 28.
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
