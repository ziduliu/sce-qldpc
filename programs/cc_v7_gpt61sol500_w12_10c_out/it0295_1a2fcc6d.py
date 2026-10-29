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
CONCEPT = ("C13:C9 with action 3, order 117, shapes (2,1)x(2,1): fresh transpose constituents selected using central-coordinate and normal-rotation repairs, block-local physical witnesses and witness-seeded full logical searches should exceed cell-0.200 fitness 113.3054; orientation alone is not an improvement claim.")
GROUP = ("Keep the metacyclic group and order byte-for-byte under L2. RUN STATE records 113.3054 on the wide order-117 bed and 111.7230 on its tall counterpart. Central C3 and commuting C9 translations provide restricted word subspaces for both verified transpose actions. These measurements motivate retaining the group, without certifying a group-distance advantage.")
SHAPES = ("Equal (2,1) sides target cell 0.200: n=585, kb=117*(1-2)*(1-2)=117 and guaranteed rate 1/5; q<=119 allows retaining order 117. Both component maps have 117 variables and 234 checks, with 234-coordinate/117-check transpose kernels. Current-parent weight-23 minima were entirely sector one, especially Acol1 across B columns and Bcol1 across A columns; slots 21 and 27 retained weight 24. Fresh tall fillings screen the corresponding Arow/Brow sector-two families. Exact complete ranks retain 0.175<=k/n<0.225.")
FILLING = ("Twenty tasks cross five balanced/skew quotient architectures with four central-coordinate policies. Four distinct terms per tall cell give row totals four, column totals eight and both CSS check weights twelve. Compile transpose columns exclusively from verified unit-cell lifts. Search unrestricted, orbit-even, invariant and single-block physical supports. Most repairs change central grades; every sixth repair also changes a normal rotation residue and therefore re-screens both restricted matrices instead of freezing the parent's quotient bottleneck. Carry only syndrome-verified surviving witnesses. Equal deeper budgets precede exact rank gates, sector-two logical embedding, stabilizer dressing and witness-seeded full-CSS searches. Uncoloured Tanner signatures omit k and conservatively reject equal signatures. Restricted exclusions and sampled readings are not quantum-distance certificates.")
FAMILY_NAME = "r020_C13semiC9_21x21_rotationgrade_blocklocal_seeded_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 117

def custom_ops(q):
    # x^13=y^9=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    if not valid_q(q):
        raise ValueError("C13:C9 requires order 117")
    els = [(a, b) for b in range(9) for a in range(13)]
    action = [pow(3, b, 13) for b in range(9)]

    def red(g):
        return (g[0] % 13, g[1] % 9)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 13, (b+d) % 9)

    def inv(g):
        a, b = red(g)
        return ((-action[(-b) % 9]*a) % 13, (-b) % 9)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)


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


# Historical Dic11 support enumerator retained but not executed.
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
    """Extract tall transpose actions from the VERIFIED complete lift."""
    assert A_SHAPE == B_SHAPE == (2, 1)
    mask = (1 << q)-1
    # Sector two is ordered (Arow,Brow), following sector one's q bits.
    # Fix Brow0 in HZ for A^T and Arow0 in HX for B^T.
    arows = [
        ((row >> q) & mask) | (((row >> (3*q)) & mask) << q)
        for row in hz[:q]]
    brows = [(row >> q) & ((1 << (2*q))-1) for row in hx[:q]]
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


def _legacy_dic_NEW_CTX(rng, G, idx):
    """Inactive historical Dic11 optimizer; superseded by NEW_CTX below."""
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

    def draw_grid(counts, model):
        if model == 0:
            return _label_grid(rng, G, counts)
        grid = []
        for i in range(2):
            row = []
            for j in range(3):
                count = counts[3*i+j]
                phase = (rng.randrange(4) + idx + i + 2*j) % 4
                if model == 1:
                    direction = rng.choice((-1, 1))
                    quota = [(phase + direction*t) % 4
                             for t in range(count)]
                elif count == 1:
                    quota = [phase]
                elif count == 2:
                    quota = [phase, (phase + rng.choice((1, 2, 3))) % 4]
                elif count == 3:
                    quota = [phase, phase,
                             (phase + rng.choice((1, 3))) % 4]
                else:
                    # Distinct occupancy experiments, not translations
                    # of the parent's single four-term quota.
                    offsets = ((0, 0, 1, 3), (0, 0, 2, 2),
                               (0, 1, 1, 2))
                    quota = [(phase+t) % 4
                             for t in offsets[(idx+i+j) % 3]]
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
                         probe(item["ac"], item["aw"], 12))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 12))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(item["da"],
                         probe(item["ac"], item["aw"], 64))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 64))
        item["logical"] = dressed_reading(item)
    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert q <= k and 8*k < n
    return {"A": A, "B": B}


from functools import lru_cache


_R2_PATTERNS = (
    (0, 1, 2, 3),
    (0, 1, 3, 6),
    (0, 0, 1, 2),
    (0, 0, 1, 4),
    (0, 1, 1, 4),
    (0, 0, 0, 1),
    (0, 1, 4, 7),
    (0, 1, 2, 4),
)

# Five different arrangements, crossed with four rotation policies.
_R2_CARDS = (
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (2, 4, 0, 6),
    (5, 6, 2, 7),
    (1, 7, 4, 5),
)


def _r2_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _r2_basis(rows):
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


def _r2_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _r2_short(cols):
    """Exact component exclusion through four, not a quantum certificate."""
    seen = {0}
    for col in cols:
        if col in seen:
            return False
        seen.add(col)
    for i, col in enumerate(cols):
        for other in cols[:i]:
            syndrome = col ^ other
            if syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _r2_fiber_seeds(cols, G):
    """True kernel words restricted to normal-C13 fiber indicators."""
    q = G.q
    fibers = [
        sum(1 << G.idx(g) for g in G.elements if g[1] == b)
        for b in range(9)]
    expanded = [fiber << (j*q) for j in range(2) for fiber in fibers]
    pivots, dependencies = {}, []
    for c, full in enumerate(expanded):
        syndrome = 0
        word = full
        while word:
            bit = word & -word
            syndrome ^= cols[bit.bit_length()-1]
            word ^= bit
        small = 1 << c
        while syndrome:
            p = syndrome.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = (syndrome, small)
                break
            syndrome ^= old[0]
            small ^= old[1]
        if not syndrome:
            full = 0
            while small:
                bit = small & -small
                full ^= expanded[bit.bit_length()-1]
                small ^= bit
            dependencies.append(full)
    seeds = set(dependencies)
    for i, word in enumerate(dependencies):
        seeds.update(word ^ other for other in dependencies[:i])
    seeds.discard(0)
    return sorted(seeds, key=lambda v: (v.bit_count(), v))[:24]


def _r2_probe(cols, seeds, rng, trials):
    """Actual nonzero kernel witnesses, retained across fresh searches."""
    width = len(cols)
    memory = sorted(set(seeds), key=int.bit_count)[:24]
    best = min((v.bit_count() for v in memory), default=width+1)
    order = list(range(width))
    for trial in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
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
                basis.append(word)
                best = min(best, word.bit_count())

        for sweep in range(1 + int(trials >= 12)):
            reducers = sorted(set(basis+memory), key=int.bit_count)[:24]
            rng.shuffle(reducers)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                basis[i] = word
                best = min(best, weight)

        short = sorted(set(basis+memory), key=int.bit_count)[:48]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    cw = combined.bit_count()
                    best = min(best, cw)
                    if cw <= best+4:
                        discoveries.append(combined)
        if trials >= 12 and trial % 4 == 0:
            small = short[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        combined = pair ^ small[h]
                        if combined:
                            best = min(best, combined.bit_count())
                            discoveries.append(combined)
        memory = sorted(set(discoveries+memory),
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 8:
            break
    seeds[:] = memory
    return best


def _r2_dressed(item, q):
    """Verified sector-two logical embeddings, retained for full searches."""
    assert A_SHAPE == B_SHAPE == (2, 1)
    hx, hz = item["hx"], item["hz"]
    mask = (1 << q)-1
    best = item["n"]+1
    item["logical_seeds"] = {"X": [], "Z": []}
    ceiling = min(item["da"], item["db"])+6
    for side, seeds, checks, stabilizers, kind in (
            ("A", item["aw"], hz, hx, "X"),
            ("B", item["bw"], hx, hz, "Z")):
        span = _r2_basis(stabilizers)
        for seed in sorted(set(seeds), key=int.bit_count)[:12]:
            if seed.bit_count() > ceiling:
                continue
            for fixed in range(2):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((1+2*j+fixed)*q) for j in range(2))
                else:
                    word = seed << ((1+2*fixed)*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _r2_outside(word, span):
                    continue
                item["logical_seeds"][kind].append(word)
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
                    nearby.sort(key=lambda entry: entry[0])
                    rows = [row for _, row in nearby[:16]]
                    for i, row in enumerate(rows):
                        for other in rows[:i]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
                item["logical_seeds"][kind].append(word)
    for kind, words in item["logical_seeds"].items():
        item["logical_seeds"][kind] = sorted(set(words), key=int.bit_count)[:24]
    return best


def _r2_signature(hx, hz, n, k):
    """Uncoloured Tanner invariant; inequality certifies nonisomorphism.

    Equal signatures are discarded conservatively, not called isomorphic.
    Check-overlap adjacency and its local triangles ignore X/Z colours.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    local = [{} for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if not overlap:
                continue
            adjacency[i] |= 1 << j
            adjacency[j] |= 1 << i
            a, b = (weights[j], overlap), (weights[i], overlap)
            local[i][a] = local[i].get(a, 0)+1
            local[j][b] = local[j].get(b, 0)+1
    profiles = []
    for i, neighbors in enumerate(adjacency):
        triangles = 0
        remaining = neighbors
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        profiles.append((weights[i], tuple(sorted(local[i].items())),
                         triangles//2))
    columns = _r2_columns(rows, n)
    # k depends on the X/Z partition, not just the uncoloured graph.
    return (n, tuple(sorted(profiles)),
            tuple(sorted(col.bit_count() for col in columns)))


@lru_cache(maxsize=1)
def _r2_bank(G):
    """Deterministic bank with twenty policies and no orbit fallbacks."""
    assert G.q == 117
    q, width = G.q, 2*G.q
    bank, emitted = [], set()

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    for slot in range(N_SLOTS):
        card, mode = divmod(slot % 20, 4)
        policy = _R2_CARDS[card]
        rng = random.Random(BASE_SEED*10007 + 11702001 + slot)

        def cell(pattern):
            for retry in range(128):
                phase = rng.randrange(9)
                direction = rng.choice((1, 2, 4, 5, 7, 8))
                bs = [(phase+direction*b) % 9
                      for b in _R2_PATTERNS[pattern]]
                if mode == 0:
                    residues = rng.sample(range(13), 4)
                elif mode == 1:
                    base, step = rng.randrange(13), rng.randrange(1, 13)
                    residues = [(base+t*step) % 13 for t in range(4)]
                elif mode == 2:
                    base = rng.randrange(13)
                    u, v = rng.sample(range(1, 13), 2)
                    residues = [base, (base+u) % 13,
                                (base+v) % 13, (base+u+v) % 13]
                else:
                    residues = [rng.randrange(13) for _ in range(4)]
                result = list(zip(residues, bs))
                if len(set(result)) != 4:
                    continue
                relatives = [G.mul(G.inv(result[0]), g)
                             for g in result[1:]]
                if not any(G.mul(g, h) != G.mul(h, g)
                           for i, g in enumerate(relatives)
                           for h in relatives[:i]):
                    continue
                # Nonidentity (0,3) and (0,6) have order three.
                if any(
                    (relative[0] == 0 and relative[1] in (3, 6))
                    for i, g in enumerate(result)
                    for h in result[:i]
                    for relative in (G.mul(G.inv(g), h),)):
                    continue
                return result
            raise ValueError("quartic occupancy recipe exhausted")

        def proposal():
            cells = [cell(pattern) for pattern in policy]
            return [[cells[0], cells[1]]], [[cells[2], cells[3]]]

        finalists, seen = [], set()
        chosen = None
        for batch in range(3):
            for attempt in range(24):
                A, B = proposal()
                labels = tuple(tuple(sorted(c))
                               for grid in (A, B) for row in grid for c in row)
                if labels in seen:
                    continue
                seen.add(labels)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (175*n <= 1000*k < 225*n and k >= q):
                    continue
                ar, br = _component_checks(hx, hz, q)
                ac = _r2_columns(ar, width)
                bc = _r2_columns(br, width)
                if not (_r2_short(ac) and _r2_short(bc)):
                    continue
                aw, bw = _r2_fiber_seeds(ac, G), _r2_fiber_seeds(bc, G)
                item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                            ac=ac, bc=bc, aw=aw, bw=bw,
                            da=_r2_probe(ac, aw, rng, 2),
                            db=_r2_probe(bc, bw, rng, 2),
                            cycles=_four_cycles(ar)+_four_cycles(br))
                finalists.append(item)
                finalists.sort(key=merit, reverse=True)
                del finalists[6:]

            for item in finalists:
                item["da"] = min(item["da"],
                    _r2_probe(item["ac"], item["aw"], rng, 12))
                item["db"] = min(item["db"],
                    _r2_probe(item["bc"], item["bw"], rng, 12))
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:4]
            for item in finalists:
                item["da"] = min(item["da"],
                    _r2_probe(item["ac"], item["aw"], rng, 32))
                item["db"] = min(item["db"],
                    _r2_probe(item["bc"], item["bw"], rng, 32))
                item["logical"] = _r2_dressed(item, q)

            for item in sorted(finalists, key=merit, reverse=True):
                signature = _r2_signature(
                    item["hx"], item["hz"], item["n"], item["k"])
                if signature in emitted:
                    continue
                assert css_ok(item["hx"], item["hz"])
                assert max(row.bit_count()
                           for row in item["hx"]+item["hz"]) == 12
                assert _grid_weights([[4, 4]], [[4, 4]]) == 12
                assert _no_dead_lines([[4, 4]], [[4, 4]])
                emitted.add(signature)
                chosen = (item["A"], item["B"])
                break
            if chosen is not None:
                break

        # Never replace an unsuccessful card with an emitted orbit copy.
        bank.append(chosen)
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure deterministic bank; idx=-1 memoizes shared verified compilation."""
    if idx >= 0:
        item = NEW_CTX(None, G, -1)[idx]
        if item is None:
            raise ValueError("no fresh norm-screened filling in target band")
        A, B = item
        return {"A": [[list(c) for c in row] for row in A],
                "B": [[list(c) for c in row] for row in B]}

    assert G.q == 117
    assert A_SHAPE == B_SHAPE == (2, 1)
    q, width = G.q, 2*G.q
    mask = (1 << q)-1
    stub = [[[G.e]], [[G.e]]]

    # Entries describe Arow0,Arow1,Brow0,Brow1. Every task optimizes
    # fresh tall voltages; no archived wide filling is transposed.
    patterns = ((0, 0, 1, 2), (0, 0, 1, 1), (0, 0, 0, 1))
    cards = (
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (0, 2, 0, 2),
        (0, 0, 0, 2),
        (0, 1, 0, 2),
    )

    def cycles(cols):
        total = 0
        for i, col in enumerate(cols):
            for other in cols[:i]:
                overlap = (col & other).bit_count()
                total += overlap*(overlap-1)//2
        return total

    # F2-linear superposition of scaffold-generated single-term blocks.
    # No regular representation is implemented in this optimizer.
    atlas, partitions = {}, {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g]], [[]]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5*q
            source = hz if side == "A" else hx
            rows = [(row >> q) & mask for row in source[:q]]
            blocks.append(tuple(_r2_columns(rows, q)))
        atlas[side] = tuple(blocks)

        for length, generator in ((3, (0, 3)), (9, (0, 1))):
            covered, orbits = set(), []
            for g in G.elements:
                if G.idx(g) in covered:
                    continue
                h, orbit = g, []
                for _ in range(length):
                    orbit.append(G.idx(h))
                    # These commuting translations define word subspaces,
                    # not an alternative parity-matrix construction.
                    h = (G.mul(h, generator) if side == "A"
                         else G.mul(generator, h))
                assert G._key(h) == G._key(g)
                assert len(set(orbit)) == length
                covered.update(orbit)
                orbits.append(tuple(sorted(orbit)))
            assert len(covered) == q
            assert len(orbits)*length == q
            partitions[side, length] = tuple(orbits)

    def component_columns(side, cells):
        """Transpose columns assembled only from verified unit contributions."""
        cols = [0]*width
        for j, row in enumerate(cells):
            for g in row[0]:
                contribution = atlas[side][G.idx(g)]
                for h, syndrome in enumerate(contribution):
                    cols[j*q+h] ^= syndrome
        return cols

    def fold(cols, side, length):
        """Exact restricted syndromes, checked constant on output orbits."""
        orbits = partitions[side, length]
        masks = [sum(1 << h for h in orbit) for orbit in orbits]
        folded, expanded = [], []
        for j in range(2):
            for orbit in orbits:
                syndrome, word = 0, 0
                for h in orbit:
                    syndrome ^= cols[j*q+h]
                    word |= 1 << (j*q+h)
                compact = 0
                for r, output_mask in enumerate(masks):
                    restricted = syndrome & output_mask
                    assert restricted in (0, output_mask)
                    if restricted:
                        compact |= 1 << r
                folded.append(compact)
                expanded.append(word)
        return folded, expanded

    def expand(word, indicators):
        full = 0
        while word:
            bit = word & -word
            full ^= indicators[bit.bit_length()-1]
            word ^= bit
        return full

    def norm9_short(cols):
        # Exactly exclude restricted words of folded weight one/two,
        # hence physical weight nine/eighteen. No larger claim is made.
        return 0 not in cols and len(set(cols)) == len(cols)

    def exact7(cols):
        """Witness through seven, or exhaustive exclusion through seven.

        Eight is an exclusion flag, never a found distance-eight word.
        After singleton/pair/triple collisions have passed, one stored
        representative per syndrome suffices for the quadruple query.
        """
        bits = [1 << i for i in range(len(cols))]
        seen = {0: 0}

        def insert(syndrome, word):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif old != word:
                witness = old ^ word
                return witness.bit_count(), witness
            return None

        for i, syndrome in enumerate(cols):
            hit = insert(syndrome, bits[i])
            if hit is not None:
                return hit
        for i in range(1, len(cols)):
            for j in range(i):
                hit = insert(cols[i] ^ cols[j], bits[i] | bits[j])
                if hit is not None:
                    return hit
        for i in range(2, len(cols)):
            for j in range(1, i):
                syndrome = cols[i] ^ cols[j]
                word = bits[i] | bits[j]
                for h in range(j):
                    hit = insert(syndrome ^ cols[h], word | bits[h])
                    if hit is not None:
                        return hit
        # All words through six have a partition into two subsets
        # of size <=3. Every weight-seven word has a 4+3 partition.
        for i in range(3, len(cols)):
            for j in range(2, i):
                syndrome = cols[i] ^ cols[j]
                for h in range(1, j):
                    triple = syndrome ^ cols[h]
                    for t in range(h):
                        old = seen.get(triple ^ cols[t])
                        if old is not None:
                            word = bits[i] | bits[j] | bits[h] | bits[t]
                            witness = word ^ old
                            assert witness and witness.bit_count() <= 7
                            return witness.bit_count(), witness
        return 8, 0

    def draw_cell(local, kind, mode, slot, column):
        for _ in range(64):
            quotient_shift = local.randrange(3)
            direction = local.choice((1, 2))
            quotient = [(quotient_shift+direction*b) % 3
                        for b in patterns[kind]]
            residues = (local.sample(range(13), 4) if mode != 3
                        else [local.randrange(13) for _ in range(4)])
            if mode == 0:
                central = [local.randrange(3) for _ in range(4)]
            elif mode == 1:
                central = [0]*4
                for r in range(3):
                    positions = [t for t, b in enumerate(quotient) if b == r]
                    phase = local.randrange(3)
                    local.shuffle(positions)
                    for number, t in enumerate(positions):
                        central[t] = (phase+number) % 3
            elif mode == 2:
                central = [(a*(a+1)+2*column+slot//20) % 3
                           for a in residues]
            else:
                central = [0, 1, 1, 2]
                local.shuffle(central)
            cell = [(a, b+3*c)
                    for a, b, c in zip(residues, quotient, central)]
            # Distinct images modulo the central C3 preserve quartic
            # folded columns and exclude central order-three pairs.
            if len({(a, b % 3) for a, b in cell}) != 4:
                continue
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if not any(G.mul(g, h) != G.mul(h, g)
                       for i, g in enumerate(relatives)
                       for h in relatives[:i]):
                continue
            assert len(set(cell)) == 4
            return cell
        raise ValueError("new central-occupancy recipe exhausted")

    def syndrome_of(word, cols):
        syndrome = 0
        while word:
            bit = word & -word
            syndrome ^= cols[bit.bit_length()-1]
            word ^= bit
        return syndrome

    # Support translations commute with the corresponding component
    # action. They construct candidate words, never parity matrices.
    translations = {}
    for side in ("A", "B"):
        maps = []
        for t in ((0, 3), (0, 6), (1, 0), (0, 1)):
            permutation = tuple(
                G.idx(G.mul(g, t) if side == "A" else G.mul(t, g))
                for g in G.elements)
            assert len(set(permutation)) == q
            maps.append(tuple(
                1 << (j*q+permutation[h])
                for j in range(2) for h in range(q)))
        translations[side] = tuple(maps)

    def physical_probe(cols, seeds, local, side, trials):
        """Kernel search in three coordinates, always using physical weight.

        Central-orbit-even coordinates have two pair generators per
        orbit. Invariant coordinates have one triple generator. Elimination
        carries physical supports, so pair-coordinate cancellations are
        handled before weighing. Every recorded word is a true kernel word.
        """
        ordinary = [(s, 1 << i) for i, s in enumerate(cols)]
        even, invariant = [], []
        for j in range(2):
            for orbit in partitions[side, 3]:
                a, b, c = [j*q+h for h in orbit]
                even.extend((
                    (cols[a] ^ cols[b], (1 << a) | (1 << b)),
                    (cols[a] ^ cols[c], (1 << a) | (1 << c))))
                invariant.append((
                    cols[a] ^ cols[b] ^ cols[c],
                    (1 << a) | (1 << b) | (1 << c)))
        # Single-block searches expose singular-cell words that broad
        # information sets can miss. Their supports remain physical.
        modes = (ordinary, even, invariant,
                 ordinary[:q], ordinary[q:])
        memory = sorted(
            {w for w in seeds if w and syndrome_of(w, cols) == 0},
            key=lambda w: (w.bit_count(), w))[:48]
        best = min((w.bit_count() for w in memory), default=width+1)

        for trial in range(trials):
            coordinates = modes[trial % len(modes)]
            order = list(range(len(coordinates)))
            local.shuffle(order)
            pivots, dependencies = {}, []
            for c in order:
                syndrome, word = coordinates[c]
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome and word:
                    dependencies.append(word)
                    best = min(best, word.bit_count())

            # Translate short witnessed supports, not cell labels.
            # Check syndromes explicitly rather than assuming a convention.
            shifted = []
            for word in sorted(set(memory+dependencies),
                               key=int.bit_count)[:12]:
                for permutation in translations[side]:
                    moved, remaining = 0, word
                    while remaining:
                        bit = remaining & -remaining
                        moved ^= permutation[bit.bit_length()-1]
                        remaining ^= bit
                    if moved and syndrome_of(moved, cols) == 0:
                        shifted.append(moved)

            words = sorted(set(memory+dependencies+shifted),
                           key=lambda w: (w.bit_count(), w))
            for sweep in range(2):
                donors = sorted(set(words), key=int.bit_count)[:40]
                changed = False
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for donor in donors:
                        proposal = word ^ donor
                        pw = proposal.bit_count()
                        if 0 < pw < weight:
                            word, weight = proposal, pw
                            changed = True
                    reduced.append(word)
                    best = min(best, weight)
                words = sorted(set(reduced),
                               key=lambda w: (w.bit_count(), w))
                if not changed:
                    break

            discoveries = set(words+memory+shifted)
            short = sorted(discoveries, key=int.bit_count)[:32]
            for i, word in enumerate(short):
                for other in short[:i]:
                    proposal = word ^ other
                    if proposal:
                        best = min(best, proposal.bit_count())
                        discoveries.add(proposal)
            if trial % 4 == 0:
                small = short[:8]
                for i, word in enumerate(small):
                    for j in range(i):
                        pair = word ^ small[j]
                        for h in range(j):
                            proposal = pair ^ small[h]
                            if proposal:
                                best = min(best, proposal.bit_count())
                                discoveries.add(proposal)
            memory = sorted(discoveries,
                            key=lambda w: (w.bit_count(), w))[:48]
            if best <= 18:
                break

        assert memory
        assert all(syndrome_of(w, cols) == 0 for w in memory)
        seeds[:] = memory
        # This is retained-witness load, not a complete minimum census.
        hits = sum(w.bit_count() == best for w in memory)
        return best, hits

    def optimize_side(local, side, kinds, mode, slot, batch):
        """Screen fresh tall supports, then repair grades and rotations."""
        raw, seen = [], set()
        quota = 192 + 32*batch + 16*(slot//20)
        offset = 0 if side == "A" else 2

        def labels(cells):
            return tuple(tuple(sorted(row[0])) for row in cells)

        def component_merit(item):
            return (item["d"], -item["hits"], int(item["flag"] == 8),
                    -item["folded_cycles"], -item["cycles"], item["rank"])

        for attempt in range(quota):
            cells = [[draw_cell(local, kind, mode, slot, offset+j)]
                     for j, kind in enumerate(kinds)]
            key = labels(cells)
            if key in seen:
                continue
            seen.add(key)
            cols = component_columns(side, cells)
            f3, e3 = fold(cols, side, 3)
            if not _r2_short(f3):
                continue
            f9, e9 = fold(cols, side, 9)
            if not norm9_short(f9):
                continue
            rank = gf2_rank(cols)
            if rank < q-4:
                continue
            item = dict(cells=cells, cols=cols, f3=f3, e3=e3,
                        f9=f9, e9=e9, rank=rank,
                        folded_cycles=cycles(f3), attempt=attempt)
            raw.append(item)
            raw.sort(key=lambda x: (
                x["folded_cycles"], -x["rank"], x["attempt"]))
            del raw[18:]

        tested, clean = [], 0
        for item in raw[:12]:
            if not _r2_short(item["cols"]):
                continue
            flag, witness = exact7(item["f3"])
            item["flag"] = flag
            seeds = _r2_fiber_seeds(item["cols"], G)
            if witness:
                seeds.append(expand(witness, item["e3"]))
            s3, s9 = [], []
            _r2_probe(item["f3"], s3, local, 16)
            _r2_probe(item["f9"], s9, local, 16)
            invariant_seeds = (
                [expand(w, item["e3"]) for w in s3]
                + [expand(w, item["e9"]) for w in s9])
            seeds.extend(invariant_seeds)
            seeds = sorted(set(seeds), key=int.bit_count)[:48]
            ordinary = _r2_probe(item["cols"], seeds, local, 4)
            physical, hits = physical_probe(
                item["cols"], seeds, local, side, 15)
            item.update(seeds=seeds, invariant_seeds=invariant_seeds,
                        d=min(ordinary, physical), hits=hits,
                        cycles=cycles(item["cols"]))
            tested.append(item)
            clean += int(flag == 8)
            if clean >= 4 and len(tested) >= 7:
                break

        tested.sort(key=component_merit, reverse=True)
        elites = tested[:6]
        if not elites:
            return []

        # Individual voltage repairs, never common cell translations.
        # Rotation moves can escape a quotient support that central-only
        # repairs cannot change; their restricted kernels are re-screened.
        for step in range(24):
            parent = local.choice(elites[:min(4, len(elites))])
            cells = [[list(row[0])] for row in parent["cells"]]
            witnesses = sorted(parent["seeds"], key=int.bit_count)[:8]
            pressure = [
                1 + sum(((w >> (j*q)) & mask).bit_count()
                        for w in witnesses)
                for j in range(2)]
            j = local.choices(range(2), weights=pressure, k=1)[0]
            rotation_move = step % 6 == 5
            count = 2 if step % 4 == 3 else 1
            for t in local.sample(range(4), count):
                a, b = cells[j][0][t]
                if rotation_move:
                    a = (a+local.randrange(1, 13)) % 13
                b = (b+3*local.choice((1, 2))) % 9
                cells[j][0][t] = (a, b)
            if any(len({(a, b % 3) for a, b in row[0]}) != 4
                   for row in cells):
                continue
            cell = cells[j][0]
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if not any(G.mul(g, h) != G.mul(h, g)
                       for t, g in enumerate(relatives)
                       for h in relatives[:t]):
                continue
            key = labels(cells)
            if key in seen:
                continue
            seen.add(key)

            cols = component_columns(side, cells)
            rank = gf2_rank(cols)
            if rank < q-4 or not _r2_short(cols):
                continue
            f3, e3 = fold(cols, side, 3)
            f9, e9 = fold(cols, side, 9)
            if rotation_move:
                if not (_r2_short(f3) and norm9_short(f9)):
                    continue
                flag, witness = exact7(f3)
                if parent["flag"] == 8 and flag != 8:
                    continue
                s3, s9 = [], []
                _r2_probe(f3, s3, local, 8)
                _r2_probe(f9, s9, local, 8)
                invariant_seeds = (
                    [expand(w, e3) for w in s3]
                    + [expand(w, e9) for w in s9])
                if witness:
                    invariant_seeds.append(expand(witness, e3))
                folded_cycles = cycles(f3)
            else:
                assert f3 == parent["f3"] and f9 == parent["f9"]
                assert e3 == parent["e3"] and e9 == parent["e9"]
                flag = parent["flag"]
                folded_cycles = parent["folded_cycles"]
                invariant_seeds = list(parent["invariant_seeds"])

            # Memory is accepted only after checking the changed matrix.
            seeds = [
                w for w in parent["seeds"]+invariant_seeds
                if w and syndrome_of(w, cols) == 0]
            seeds.extend(_r2_fiber_seeds(cols, G))
            seeds = sorted(set(seeds), key=int.bit_count)[:48]
            ordinary = _r2_probe(cols, seeds, local, 4)
            physical, hits = physical_probe(cols, seeds, local, side, 10)
            child = dict(
                cells=cells, cols=cols, f3=f3, e3=e3, f9=f9, e9=e9,
                flag=flag, folded_cycles=folded_cycles,
                invariant_seeds=invariant_seeds,
                rank=rank, seeds=seeds, d=min(ordinary, physical),
                hits=hits, cycles=cycles(cols), attempt=quota+step)
            elites.append(child)
            elites.sort(key=component_merit, reverse=True)
            del elites[6:]

        # Every retained matrix gets the same fresh deeper budgets.
        # Previously found witnesses remain valid and can only lower its
        # reading. A high result is still a bounded-search upper estimate.
        for item in elites:
            ordinary = _r2_probe(
                item["cols"], item["seeds"], local, 64)
            physical, hits = physical_probe(
                item["cols"], item["seeds"], local, side, 50)
            item["d"] = min(item["d"], ordinary, physical)
            item["hits"] = sum(
                w.bit_count() == item["d"] for w in item["seeds"])
        elites.sort(key=component_merit, reverse=True)
        return elites[:3]

    def full_logical_probe(checks, stabilizers, n, local, trials, seeds=()):
        """Retain verified logical witnesses across full information sets."""
        cols = _r2_columns(checks, n)
        span = _r2_basis(stabilizers)
        memory = sorted(
            {w for w in seeds if w and syndrome_of(w, cols) == 0
             and _r2_outside(w, span)}, key=int.bit_count)[:24]
        best = n+1
        discovered = []
        order = list(range(n))

        def observe(word):
            nonlocal best
            if not word or not _r2_outside(word, span):
                return
            weight = word.bit_count()
            # Dress only by actual stabilizers, preserving logical class.
            for _ in range(2):
                changed = False
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if tw < weight:
                        word, weight = trial, tw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
            discovered.append(word)

        for word in memory:
            observe(word)
        memory = sorted(set(memory+discovered), key=int.bit_count)[:24]

        for _ in range(trials):
            discovered = []
            local.shuffle(order)
            pivots, words = {}, []
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
                    words.append(word)
            # Keep logical directions separately from short stabilizers;
            # otherwise a short-word list can contain only check rows.
            logical = sorted(
                {w for w in words+memory if _r2_outside(w, span)},
                key=int.bit_count)[:16]
            reducers = sorted(set(words), key=int.bit_count)[:32] + memory[:16]
            for start in logical:
                word = start
                for _ in range(2):
                    weight = word.bit_count()
                    chosen = word
                    for reducer in reducers:
                        trial = word ^ reducer
                        tw = trial.bit_count()
                        if trial and tw < weight and _r2_outside(trial, span):
                            chosen, weight = trial, tw
                    if chosen == word:
                        break
                    word = chosen
                observe(word)
            for i, word in enumerate(logical):
                for other in logical[:i]:
                    observe(word ^ other)
            memory = sorted(set(memory+logical+discovered),
                            key=int.bit_count)[:24]
        return best

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1),
                item.get("full", width+1))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        card, mode = divmod(slot % 20, 4)
        policy = cards[card]
        local = random.Random(BASE_SEED*100003 + 11724193 + 104729*slot)
        chosen = None
        for batch in range(2):
            ap = optimize_side(local, "A", policy[:2], mode, slot, batch)
            bp = optimize_side(local, "B", policy[2:], mode, slot, batch)
            assembled = []
            for a in ap:
                for b in bp:
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (175*n <= 1000*k < 225*n and k >= q):
                        continue
                    # Verify compiled components against a complete lift.
                    ar, br = _component_checks(hx, hz, q)
                    assert _r2_columns(ar, width) == a["cols"]
                    assert _r2_columns(br, width) == b["cols"]
                    if not (_r2_short(_r2_columns(hx, n))
                            and _r2_short(_r2_columns(hz, n))):
                        continue
                    assembled.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        ac=a["cols"], bc=b["cols"],
                        aw=list(a["seeds"]), bw=list(b["seeds"]),
                        da=a["d"], db=b["d"],
                        cycles=a["cycles"]+b["cycles"]))

            finalists = sorted(assembled, key=merit, reverse=True)[:3]
            for item in finalists:
                item["da"] = min(item["da"], _r2_probe(
                    item["ac"], item["aw"], local, 32))
                item["db"] = min(item["db"], _r2_probe(
                    item["bc"], item["bw"], local, 32))
                item["logical"] = _r2_dressed(item, q)
                dx = full_logical_probe(
                    item["hz"], item["hx"], item["n"], local, 8,
                    seeds=item["logical_seeds"]["X"])
                dz = full_logical_probe(
                    item["hx"], item["hz"], item["n"], local, 8,
                    seeds=item["logical_seeds"]["Z"])
                item["full"] = min(dx, dz)

            for item in sorted(finalists, key=merit, reverse=True):
                signature = _r2_signature(
                    item["hx"], item["hz"], item["n"], item["k"])
                if signature in emitted:
                    continue
                assert css_ok(item["hx"], item["hz"])
                assert max(row.bit_count()
                           for row in item["hx"]+item["hz"]) == 12
                sa = [[len(cell) for cell in row] for row in item["A"]]
                sb = [[len(cell) for cell in row] for row in item["B"]]
                assert sa == sb == [[4], [4]]
                assert _grid_weights(sa, sb) == 12
                assert _no_dead_lines(sa, sb)
                assert all(len(cell) == len(set(cell)) == 4
                           for M in (item["A"], item["B"])
                           for row in M for cell in row)
                emitted.add(signature)
                chosen = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in M)
                    for M in (item["A"], item["B"]))
                break
            if chosen is not None:
                break
        # An unsuccessful policy is not replaced by an orbit copy.
        bank.append(chosen)
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return every freshly optimized tall cell without further sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: The Dic11 parent's lightest logicals are always sector-one-only; measured slot 0 contained two weight-24 X logicals touching both sectors, including Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Noncommuting quartics, central order-three pair exclusion and the existing C13-fiber probes remove weight-nine/eighteen witnesses on C13:C9; current-parent slots 5,6,25,26,45,46 still read d2s=9, numerous slots read eighteen, and measured slot 28 falls to d30s=22.
# REFUTED: C3/C9 invariant screens plus the existing ordinary and dressed probes ensure selected d2s=24 codes retain d30s>=24; current-parent slot 0 fell to 21 and slot 46 to 23, with minima entirely in sector 1.
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
