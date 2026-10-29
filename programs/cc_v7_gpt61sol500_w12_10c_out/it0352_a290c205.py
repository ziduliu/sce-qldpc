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
CONCEPT = ("C13 semidirect C9 with action 3, order 117, shapes (1,2)x(1,2): cycle-directed individual voltage repairs and correlated physical-support information sets should exceed cell-0.200 frontier fitness 113.3054; measured searches can refute this.")
GROUP = ("Frozen L1 group and order, kept byte-for-byte. RUN STATE records 113.3 on this wide bed and 111.7 on its tall counterpart. Central C3 and commuting order-nine translations provide complementary restricted word subspaces. All parity actions, syndromes and assembled ranks use verified construction helpers; these measurements do not certify a group-distance advantage.")
SHAPES = ("Frozen equal (1,2) sides for cell 0.200: n=585, kb=117, guaranteed rate 1/5, equal 234-variable/117-check components, and maximum admissible lift order 119. The group fixes q=117. All five current-parent measured readings were 23, entirely in sector one, prominently Acol1 across both B columns for X and Bcol1 across both A columns for Z. Exact complete ranks retain 0.175<=k/n<0.225.")
FILLING = ("Twenty tasks cross new quotient architectures with four central-grade correlation rules, always four distinct terms per cell and check weight twelve. Exclude zero normal-C13 projections and order-nine-invariant words through physical weight eighteen. Prefer fewer ordinary four-cycles, then fewer six-cycles where four-cycle-free, and repair individual rotation residues or central grades before the inherited central-norm-through-seven sieve. Search ordinary and central-even coordinates with physical-support propagation, witness-biased information sets, broader pair/triple recombination and syndrome-verified central translations. Equal deeper budgets precede verified component comparisons, exact full ranks, logical dressing and uncoloured Tanner discrimination without k. Restricted exclusions and sampled readings are not quantum-distance certificates.")
FAMILY_NAME = "r020_C13semiC9_12x12_cyclegeometry_correlated_isd_v3"

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

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


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
    """Extract both component actions from the VERIFIED full lift."""
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
    """Syndrome- and stabilizer-tested logical embeddings and dressings."""
    hx, hz = item["hx"], item["hz"]
    mask = (1 << q)-1
    best = item["n"]+1
    ceiling = min(item["da"], item["db"])+6
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        span = _r2_basis(stabilizers)
        for seed in sorted(set(seeds), key=int.bit_count)[:6]:
            if seed.bit_count() > ceiling:
                continue
            for fixed in range(2):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((2*j+fixed)*q) for j in range(2))
                else:
                    word = seed << (fixed*2*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _r2_outside(word, span):
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
                    nearby.sort(key=lambda entry: entry[0])
                    rows = [row for _, row in nearby[:10]]
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
    return (n, k, tuple(sorted(profiles)),
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


@lru_cache(maxsize=None)
def NEW_CTX(rng, G, idx):
    """Pure deterministic component optimization, memoized across slots.

    idx=-1 is an internal immutable atlas context, never an emitted slot.
    Ordinary calls canonicalize away the harness RNG; all searches use
    fresh deterministic seeds belonging to this new filling model.
    """
    if rng is not None:
        return NEW_CTX(None, G, idx)
    assert G.q == 117
    q, width = G.q, 2*G.q

    if idx == -1:
        # Central translations commute with BOTH verified component
        # actions. These are orbit partitions, not a hand-written lift.
        covered, orbits = set(), []
        z = (0, 3)
        for g in G.elements:
            if G.idx(g) in covered:
                continue
            h = G.mul(g, z)
            t = G.mul(h, z)
            orbit = tuple(sorted((G.idx(g), G.idx(h), G.idx(t))))
            assert len(set(orbit)) == 3
            assert G.mul(t, z) == g
            covered.update(orbit)
            orbits.append(orbit)
        orbits = tuple(orbits)
        assert len(orbits) == 39 and len(covered) == q
        masks = tuple(sum(1 << h for h in orbit) for orbit in orbits)

        def fold_unit(columns):
            folded = []
            for orbit in orbits:
                syndrome = 0
                for h in orbit:
                    syndrome ^= columns[h]
                compact = 0
                for r, (output, mask) in enumerate(zip(orbits, masks)):
                    restricted = syndrome & mask
                    assert restricted == 0 or restricted == mask
                    if restricted:
                        compact |= 1 << r
                folded.append(compact)
            return tuple(folded)

        # F2-linear superposition of scaffold-built unit contributions.
        # No alternate regular-representation convention is implemented.
        stub = [[[G.e], [G.e]]]
        mask = (1 << q)-1
        atlas = {}
        for side in ("A", "B"):
            ordinary, folded = [], []
            for g in G.elements:
                unit = [[[g], []]]
                A, B = (unit, stub) if side == "A" else (stub, unit)
                hx, hz, _ = lifted_matrices(G, A, B)
                source = hx if side == "A" else hz
                rows = [row & mask for row in source[:q]]
                columns = tuple(_r2_columns(rows, q))
                ordinary.append(columns)
                folded.append(fold_unit(columns))
            atlas[side] = (tuple(ordinary), tuple(folded))
        # Commuting support translations define restricted word spaces.
        # Their syndromes will still be obtained from verified columns.
        orbit9 = {}
        for side in ("A", "B"):
            covered9, groups9 = set(), []
            for g in G.elements:
                if G.idx(g) in covered9:
                    continue
                h, orbit = g, []
                for _ in range(9):
                    orbit.append(G.idx(h))
                    h = (G.mul(h, (0, 1)) if side == "A"
                         else G.mul((0, 1), h))
                assert h == g and len(set(orbit)) == 9
                assert not covered9.intersection(orbit)
                covered9.update(orbit)
                groups9.append(tuple(sorted(orbit)))
            assert len(groups9) == 13 and len(covered9) == q
            orbit9[side] = tuple(groups9)
        return {"atlas": atlas, "orbits": orbits,
                "orbit9": orbit9, "signatures": ()}

    previous = NEW_CTX(None, G, idx-1)
    atlas, orbits = previous["atlas"], previous["orbits"]
    orbit9 = previous["orbit9"]
    emitted = set(previous["signatures"])
    search_rng = random.Random(
        BASE_SEED*100003 + 11733157 + 104729*idx)

    # Five architectures, each crossed with four CENTRAL-coordinate
    # policies. Repeated policies for additional slots optimize fresh
    # labels; no stored filling is transported or relabeled.
    cards = (
        (0, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 1),
        (0, 2, 0, 1),
        (1, 2, 0, 2),
    )
    patterns = ((0, 0, 1, 2), (0, 0, 0, 1), (0, 0, 1, 1))
    architecture, mode = divmod(idx % 20, 4)
    card = cards[architecture]

    def draw_cell(kind, location):
        for _ in range(96):
            phase = search_rng.randrange(3)
            direction = search_rng.choice((1, 2))
            residues = [(phase+direction*r) % 3
                        for r in patterns[kind]]
            rotations, central = [None]*4, [None]*4
            global_phase = search_rng.randrange(3)
            for residue in range(3):
                positions = [t for t, r in enumerate(residues)
                             if r == residue]
                values = search_rng.sample(range(13), len(positions))
                local_phase = search_rng.randrange(3)
                for number, (t, a) in enumerate(zip(positions, values)):
                    rotations[t] = a
                    if mode == 0:
                        # Repeated quotient cosets span different
                        # central grades whenever their counts allow.
                        c = (local_phase+number) % 3
                    elif mode == 1:
                        # Nonlinear normal-coordinate/grade correlation,
                        # not a common automorphism of another task.
                        c = (local_phase +
                             (a*a+(location+1+idx//20)*a) % 13) % 3
                    elif mode == 2:
                        # Couple the normal residue and quotient position
                        # across cosets rather than translating all grades.
                        c = (global_phase+t*t+location*residue+a) % 3
                    else:
                        c = search_rng.randrange(3)
                    central[t] = c
            cell = [(a, r+3*c) for a, r, c
                    in zip(rotations, residues, central)]
            # Distinct quotient projections also exclude pairs whose
            # relative voltage is a nonidentity central order-three term.
            assert len({(a, b % 3) for a, b in cell}) == 4
            projection = 0
            for a, b in cell:
                projection ^= 1 << b
            if not projection:
                # A zero C9 projection gives a normal-C13-constant
                # single-cell component kernel word of physical weight 13.
                continue
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if any(G.mul(g, h) != G.mul(h, g)
                   for t, g in enumerate(relatives)
                   for h in relatives[:t]):
                return cell
        raise ValueError("noncommuting central-coordinate recipe exhausted")

    def compose(side, cells):
        ordinary, folded = atlas[side]
        cols, norm = [0]*width, [0]*78
        for j, cell in enumerate(cells[0]):
            for g in cell:
                gi = G.idx(g)
                for h, syndrome in enumerate(ordinary[gi]):
                    cols[j*q+h] ^= syndrome
                for h, syndrome in enumerate(folded[gi]):
                    norm[j*39+h] ^= syndrome
        return cols, norm

    def restricted_witness(cols, through_seven):
        """Exact existence test on the central-invariant subspace.

        None means exclusion through six or seven, NOT a witness at the
        next weight. A returned support is a true folded kernel word.
        """
        bits = [1 << i for i in range(len(cols))]
        seen = {0: 0}

        def insert(syndrome, support):
            old = seen.get(syndrome)
            if old is not None:
                difference = support ^ old
                if difference:
                    return difference
            else:
                seen[syndrome] = support
            return None

        for i, ci in enumerate(cols):
            hit = insert(ci, bits[i])
            if hit is not None:
                return hit
        for i, ci in enumerate(cols):
            for j in range(i):
                hit = insert(ci ^ cols[j], bits[i] | bits[j])
                if hit is not None:
                    return hit
        for i, ci in enumerate(cols):
            for j in range(i):
                pair, support = ci ^ cols[j], bits[i] | bits[j]
                for h in range(j):
                    hit = insert(pair ^ cols[h], support | bits[h])
                    if hit is not None:
                        return hit
        if not through_seven:
            return None

        # Every remaining support through seven admits a 4+<=3 split.
        for i, ci in enumerate(cols):
            for j in range(i):
                pair = ci ^ cols[j]
                for h in range(j):
                    triple = pair ^ cols[h]
                    for t in range(h):
                        old = seen.get(triple ^ cols[t])
                        if old is not None:
                            return (bits[i] | bits[j] | bits[h]
                                    | bits[t]) ^ old
        return None

    expanded = tuple(
        sum(1 << (j*q+h) for h in orbit)
        for j in range(2) for orbit in orbits)

    def expand(word):
        full = 0
        while word:
            bit = word & -word
            full ^= expanded[bit.bit_length()-1]
            word ^= bit
        return full

    def syndrome_of(word, cols):
        syndrome = 0
        while word:
            bit = word & -word
            syndrome ^= cols[bit.bit_length()-1]
            word ^= bit
        return syndrome

    def norm9_columns(cols, side):
        """Exact syndromes of 26 disjoint physical nine-point indicators."""
        result = []
        for block in range(2):
            for orbit in orbit9[side]:
                syndrome = 0
                for h in orbit:
                    syndrome ^= cols[block*q+h]
                result.append(syndrome)
        return result

    def cycle_profile(cols):
        """Four-cycles; exact six-cycles only in the four-cycle-free case."""
        four = _four_cycles(_r2_columns(cols, q))
        if four:
            return four, 0
        adjacency = [0]*q
        internal_triangles = 0
        for syndrome in cols:
            neighbors = []
            remaining = syndrome
            while remaining:
                bit = remaining & -remaining
                neighbors.append(bit.bit_length()-1)
                remaining ^= bit
            degree = len(neighbors)
            internal_triangles += degree*(degree-1)*(degree-2)//6
            for a in neighbors:
                for b in neighbors:
                    if a != b:
                        adjacency[a] |= 1 << b
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            remaining = neighbors >> (a+1)
            while remaining:
                bit = remaining & -remaining
                b = a+1+bit.bit_length()-1
                triangles += ((neighbors & adjacency[b]) >> (b+1)).bit_count()
                remaining ^= bit
        # No pair of checks shares two variables after the four-cycle
        # exclusion. Other check triangles are exactly Tanner six-cycles.
        six = triangles-internal_triangles
        assert six >= 0
        return four, six

    # Central translations of SUPPORTS, never transport of cell labels.
    central_move = [0]*width
    for block in range(2):
        for orbit in orbits:
            for t, h in enumerate(orbit):
                central_move[block*q+h] = (
                    1 << (block*q+orbit[(t+1) % 3]))

    def correlated_probe(cols, seeds, trials):
        """Found kernel words using correlated physical information sets.

        Ordinary coordinates have weight one; central-even coordinates
        have weight two. Elimination carries physical words, so coordinate
        cancellations are resolved before scoring. Warm information sets
        and recombination are heuristics, not exclusion certificates.
        """
        ordinary = [(s, 1 << c) for c, s in enumerate(cols)]
        even = []
        for block in range(2):
            for orbit in orbits:
                a, b, c = [block*q+h for h in orbit]
                even.extend((
                    (cols[a] ^ cols[b], (1 << a) | (1 << b)),
                    (cols[a] ^ cols[c], (1 << a) | (1 << c))))
        memory = sorted(
            {w for w in seeds if w and syndrome_of(w, cols) == 0},
            key=lambda w: (w.bit_count(), w))[:40]
        best = min((w.bit_count() for w in memory), default=width+1)

        for trial in range(trials):
            coordinates = even if trial % 4 == 2 else ordinary
            order = list(range(len(coordinates)))
            search_rng.shuffle(order)
            if memory and trial % 4 == 1:
                focus = search_rng.choice(memory[:min(8, len(memory))])
                # Favor pivot coordinates overlapping a witnessed support.
                # This changes the information set, not the parity matrix.
                order.sort(key=lambda t:
                           -int(bool(coordinates[t][1] & focus)))
            elif trial % 4 == 3:
                # Put one physical block first, exposing singular-cell
                # dependencies before the other block fills the pivot span.
                preferred = search_rng.randrange(2)
                order.sort(key=lambda t:
                           int((t//q) != preferred))

            pivots, dependencies = {}, []
            for t in order:
                syndrome, word = coordinates[t]
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

            shifted = []
            for start in memory[:8]:
                moved = start
                for _ in range(2):
                    translated, remaining = 0, moved
                    while remaining:
                        bit = remaining & -remaining
                        translated ^= central_move[bit.bit_length()-1]
                        remaining ^= bit
                    moved = translated
                    if moved and syndrome_of(moved, cols) == 0:
                        shifted.append(moved)

            words = sorted(set(dependencies+memory+shifted),
                           key=lambda w: (w.bit_count(), w))
            for sweep in range(2):
                donors = sorted(set(words+memory),
                                key=lambda w: (w.bit_count(), w))[:48]
                search_rng.shuffle(donors)
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for donor in donors:
                        proposal = word ^ donor
                        pw = proposal.bit_count()
                        if 0 < pw < weight:
                            word, weight = proposal, pw
                    reduced.append(word)
                    best = min(best, weight)
                words = sorted(set(reduced),
                               key=lambda w: (w.bit_count(), w))

            discoveries = set(words+memory+shifted)
            short = sorted(discoveries,
                           key=lambda w: (w.bit_count(), w))[:80]
            for i, word in enumerate(short):
                for other in short[:i]:
                    proposal = word ^ other
                    if proposal:
                        pw = proposal.bit_count()
                        best = min(best, pw)
                        if pw <= best+4:
                            discoveries.add(proposal)
            # Broader systematic dependency triples than the inherited
            # ten-word occasional recombination. Every result is a word,
            # not a large score inferred from failure to find one.
            small = short[:36]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        proposal = pair ^ small[h]
                        if proposal:
                            pw = proposal.bit_count()
                            best = min(best, pw)
                            if pw <= best+4:
                                discoveries.add(proposal)
            memory = sorted(discoveries,
                            key=lambda w: (w.bit_count(), w))[:40]
            if best <= 18:
                break

        assert memory and best <= width
        assert all(syndrome_of(w, cols) == 0 for w in memory)
        seeds[:] = memory
        return best

    def component_merit(item):
        return (item["d"], -item["load"], -item["cycles"],
                -item["six"], item["rank"])

    def component_pool(side, retry):
        offset = 0 if side == "A" else 2
        proposals, labels_seen = [], set()
        quota = 192 + 32*(idx//20) + 64*retry

        def admit_component(cells):
            labels = tuple(tuple(sorted(c)) for c in cells[0])
            if labels in labels_seen:
                return
            labels_seen.add(labels)
            for cell in cells[0]:
                if len(cell) != 4 or len({(a, b % 3) for a, b in cell}) != 4:
                    return
                projection = 0
                for a, b in cell:
                    projection ^= 1 << b
                if not projection:
                    return
                relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                if not any(G.mul(g, h) != G.mul(h, g)
                           for t, g in enumerate(relatives)
                           for h in relatives[:t]):
                    return
            cols, norm = compose(side, cells)
            rank = gf2_rank(cols)
            if rank < q-4 or not _r2_short(norm):
                return
            nine = norm9_columns(cols, side)
            # Nonzero distinct restricted columns exactly exclude
            # physical nine/eighteen-point invariant kernel words.
            if 0 in nine or len(set(nine)) != len(nine):
                return
            cycles, six = cycle_profile(cols)
            norm_cycles = _four_cycles(_r2_columns(norm, 39))
            item = dict(cells=cells, cols=cols, norm=norm,
                        rank=rank, cycles=cycles, six=six,
                        norm_cycles=norm_cycles,
                        quick=(cycles, norm_cycles, six, -rank,
                               len(labels_seen)))
            proposals.append(item)
            proposals.sort(key=lambda x: x["quick"])
            del proposals[24:]

        for attempt in range(quota):
            admit_component([[draw_cell(card[offset+j], offset+j)
                              for j in range(2)]])

        # Escape poor cycle geometry by changing individual voltages,
        # never by applying a common relabeling to an existing design.
        # Rotation repairs renew every restricted syndrome below.
        for step in range(48):
            if not proposals:
                admit_component([[draw_cell(card[offset+j], offset+j)
                                  for j in range(2)]])
                continue
            parent = search_rng.choice(proposals[:min(8, len(proposals))])
            cells = [[list(cell) for cell in parent["cells"][0]]]
            j = search_rng.randrange(2)
            count = 2 if step % 5 == 4 else 1
            for t in search_rng.sample(range(4), count):
                a, b = cells[0][j][t]
                if step % 3 == 2:
                    a = (a+search_rng.randrange(1, 13)) % 13
                else:
                    b = (b+3*search_rng.choice((1, 2))) % 9
                cells[0][j][t] = (a, b)
            admit_component(cells)

        tested = []
        for item in proposals:
            if not _r2_short(item["cols"]):
                continue
            hit = restricted_witness(item["norm"], False)
            item["hit"] = hit
            item["six_clean"] = hit is None
            tested.append(item)
        tested.sort(key=lambda x: (
            x["six_clean"], -x["cycles"], -x["norm_cycles"],
            -x["six"], x["rank"]), reverse=True)

        # Prefer exact seven-clean components; retain witnessed reserves
        # rather than voiding a policy when the restricted sieve is harsh.
        clean, reserve = [], []
        for item in tested[:12]:
            hit = item["hit"]
            if hit is None:
                hit = restricted_witness(item["norm"], True)
            item["hit"] = hit
            if hit is None:
                clean.append(item)
            else:
                reserve.append(item)
            if len(clean) == 4:
                break
        reserve.sort(key=lambda x: (
            x["hit"].bit_count(), -x["cycles"], x["rank"]),
            reverse=True)
        finalists = clean if clean else reserve[:6]

        ranked = []
        for item in finalists:
            folded_words = [] if item["hit"] is None else [item["hit"]]
            folded_d = _r2_probe(
                item["norm"], folded_words, search_rng, 16)
            seeds = _r2_fiber_seeds(item["cols"], G)
            seeds.extend(expand(word) for word in folded_words)
            ordinary_d = _r2_probe(
                item["cols"], seeds, search_rng, 12)
            correlated_d = correlated_probe(item["cols"], seeds, 6)
            d = min(ordinary_d, correlated_d, 3*folded_d)
            item.update(seeds=seeds, folded_words=folded_words,
                        ordinary_d=min(ordinary_d, correlated_d),
                        folded_d=folded_d, d=d,
                        load=sum(w.bit_count() <= d+1 for w in seeds))
            ranked.append(item)

        ranked.sort(key=component_merit, reverse=True)
        ranked = ranked[:4]
        for item in ranked:
            item["folded_d"] = min(item["folded_d"], _r2_probe(
                item["norm"], item["folded_words"], search_rng, 32))
            item["seeds"].extend(
                expand(word) for word in item["folded_words"])
            item["ordinary_d"] = min(item["ordinary_d"], _r2_probe(
                item["cols"], item["seeds"], search_rng, 64))
            item["ordinary_d"] = min(item["ordinary_d"],
                correlated_probe(item["cols"], item["seeds"], 24))
            item["d"] = min(item["ordinary_d"], 3*item["folded_d"])
            item["load"] = sum(
                w.bit_count() <= item["d"]+1 for w in item["seeds"])
        ranked.sort(key=component_merit, reverse=True)
        return ranked[:3]

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    chosen = None
    for retry in range(3):
        ap, bp = component_pool("A", retry), component_pool("B", retry)
        combinations = []
        for a in ap:
            for b in bp:
                A, B = a["cells"], b["cells"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (175*n <= 1000*k < 225*n and k >= q):
                    continue
                ar, br = _component_checks(hx, hz, q)
                assert _r2_columns(ar, width) == a["cols"]
                assert _r2_columns(br, width) == b["cols"]
                combinations.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    aw=list(a["seeds"]), bw=list(b["seeds"]),
                    da=a["d"], db=b["d"],
                    cycles=a["cycles"]+b["cycles"]))
        combinations.sort(key=merit, reverse=True)
        finalists = combinations[:4]
        for item in finalists:
            # Actual full-code embeddings are syndrome checked and
            # tested outside the stabilizer span before any dressing.
            item["logical"] = _r2_dressed(item, q)
        finalists.sort(key=merit, reverse=True)
        for item in finalists:
            raw_signature = _r2_signature(
                item["hx"], item["hz"], item["n"], item["k"])
            # k depends on the CSS partition, not just the uncoloured
            # Tanner graph. Omit it from this conservative discriminator.
            signature = (raw_signature[0], raw_signature[2],
                         raw_signature[3])
            if signature in emitted:
                continue
            assert css_ok(item["hx"], item["hz"])
            assert _grid_weights([[4, 4]], [[4, 4]]) == 12
            assert _no_dead_lines([[4, 4]], [[4, 4]])
            assert max(row.bit_count()
                       for row in item["hx"]+item["hz"]) == 12
            assert all(len(c) == len(set(c)) == 4
                       for grid in (item["A"], item["B"])
                       for row in grid for c in row)
            chosen = item, signature
            break
        if chosen is not None:
            break

    if chosen is None:
        raise ValueError("no fresh central-norm screened target-rate filling")
    item, signature = chosen
    return {
        "A": tuple(tuple(tuple(c) for c in row) for row in item["A"]),
        "B": tuple(tuple(tuple(c) for c in row) for row in item["B"]),
        "atlas": atlas, "orbits": orbits, "orbit9": orbit9,
        "signatures": previous["signatures"]+(signature,),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the cycle-repaired, correlated-search-selected full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: The Dic11 parent's lightest logicals are always sector-one-only; measured slot 0 contained two weight-24 X logicals touching both sectors, including Arow0/Brow0 or Arow0/Brow1.
# REFUTED: A C13:C9 quartic finalist with d2s>=23 necessarily retains that reading at 30 seconds; measured slot 28 fell from 23 to 22, and slot 0 fell from 24 to 23, with sector-1 minima.
# REFUTED: Central-norm-through-seven preference and independent ordinary component rescreening suffice for five measured readings at least 24 on this bed; current-parent slots 6,8,12,14,19 all measured d30s=23 with k=121 and sector-one-only minima.
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
