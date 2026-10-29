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
CONCEPT = ("C21 semidirect C4 with inversion, order 84, shapes (1,2)x(1,3): independently repairing the weaker cubic component and selecting on verified full-code logical witnesses should exceed cell-0.300 five-code fitness 43.1; measured searches can refute this.")
GROUP = ("L2 size-driven order repair within the unchanged cyclic-by-cyclic semidirect rule: m=21, r=4, s=20, with s^4=1 modulo 21. This group has centre two and 24 conjugacy classes, matching the invariants of the related order-84 (1,2)x(1,3) bed that scored 43.1. Matching these invariants alone does not prove group identity or distance superiority. All matrix actions and ranks use verified construction helpers.")
SHAPES = ("Redirect to cell 0.300 with unequal (1,2)x(1,3) sides: n=588, kb=84*(2-1)*(3-1)=168 and guaranteed rate 2/7. The size window permits q<=85. Components have 168 and 252 variables with 84 checks each. Current-parent measured minima occupied sector-one Acol/Bcol blocks, predominantly Acol1 across B columns and Bcol1 across A columns, without Arow/Brow participation. Exact complete ranks retain 0.275<=k/n<0.325.")
FILLING = ("Twenty deliberate tasks cross five normal-rotation pair-order architectures with four third-term constraints. Every cell is cubic, giving HX weight nine and HZ weight twelve. Compile component actions only from verified unit lifts; exclude component dependencies through four, preserve rank strata, repair witness-heavy cells and independently deepen both constituents. Verify all logical embeddings on complete matrices, search mixed-sector information sets, dress only with actual stabilizers, and retain logical directions separately from short stabilizers. Unequal uncoloured Tanner signatures distinguish emitted codes; sampled readings remain upper bounds, not distance certificates.")
FAMILY_NAME = "r030_metacyclic_C21semiC4_12x13_cubic_weakside_full_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Same metacyclic rule: x^21=y^4=1, y*x*y^-1=x^20.
    if not valid_q(q):
        raise ValueError("C21:C4 requires order 84")
    m, r, s = 21, 4, 20
    els = [(a, b) for b in range(r) for a in range(m)]
    action = [pow(s, b, m) for b in range(r)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-action[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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
    # The X/Z partition affects k; it is not an uncoloured graph invariant.
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


# Fresh asymmetric policies. Historical quartic optimizers are inactive.
_C84_CARDS = (
    ((1, 1), (1, 1, 1)),
    ((1, 2), (1, 1, 2)),
    ((2, 2), (1, 2, 2)),
    ((1, 3), (1, 2, 3)),
    ((2, 3), (2, 2, 3)),
)


def _c84_rows(cols, height):
    rows = [0]*height
    for c, original in enumerate(cols):
        value = original
        while value:
            bit = value & -value
            rows[bit.bit_length()-1] |= 1 << c
            value ^= bit
    return rows


def _c84_syndrome(word, cols):
    value = 0
    while word:
        bit = word & -word
        value ^= cols[bit.bit_length()-1]
        word ^= bit
    return value


def _c84_low4(cols, stabilizers):
    """Exact logical existence through four, modulo stabilizer row space."""
    span = _r2_basis(stabilizers)
    seen = {0: 0}
    for i, value in enumerate(cols):
        word = 1 << i
        old = seen.get(value)
        if old is None:
            seen[value] = word
        elif _r2_outside(word ^ old, span):
            return True
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _r2_outside(word ^ old, span):
                return True
    return False


def _c84_full(checks, stabilizers, n, rng, trials, seeds=()):
    """True logical witnesses, including both sectors and check dressings."""
    cols = _r2_columns(checks, n)
    span = _r2_basis(stabilizers)
    memory = sorted(
        {w for w in seeds if w and _c84_syndrome(w, cols) == 0
         and _r2_outside(w, span)},
        key=lambda w: (w.bit_count(), w))[:24]
    best = n+1
    order = list(range(n))

    def observe(word, dress=False):
        nonlocal best
        if not word or not _r2_outside(word, span):
            return None
        assert _c84_syndrome(word, cols) == 0
        if dress:
            for _ in range(3):
                weight, chosen = word.bit_count(), word
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda entry: entry[0])[:14]
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                for a, (_, row) in enumerate(nearby):
                    for _, other in nearby[:a]:
                        moved = word ^ row ^ other
                        if moved.bit_count() < weight:
                            chosen, weight = moved, moved.bit_count()
                if chosen == word:
                    break
                word = chosen
        best = min(best, word.bit_count())
        return word

    dressed = [observe(w, True) for w in memory]
    memory = sorted(set(memory+dressed), key=int.bit_count)[:24]
    split = A_SHAPE[1]*B_SHAPE[1]*84
    for trial in range(trials):
        rng.shuffle(order)
        if trial % 3 != 2:
            order.sort(key=lambda c:
                       c < split if trial % 3 == 0 else c >= split)
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

        # Do not let a short stabilizer beam erase logical directions.
        logical = sorted(
            {w for w in dependencies+memory if _r2_outside(w, span)},
            key=lambda w: (w.bit_count(), w))[:16]
        reducers = sorted(set(dependencies), key=int.bit_count)[:36]
        discoveries = set(memory)
        for start in logical:
            word = start
            for _ in range(3):
                weight, chosen = word.bit_count(), word
                for reducer in reducers+memory:
                    moved = word ^ reducer
                    if (moved and moved.bit_count() < weight
                            and _r2_outside(moved, span)):
                        chosen, weight = moved, moved.bit_count()
                if chosen == word:
                    break
                word = chosen
            hit = observe(word, True)
            if hit is not None:
                discoveries.add(hit)
        for a, word in enumerate(logical):
            for other in logical[:a]:
                hit = observe(word ^ other)
                if hit is not None:
                    discoveries.add(hit)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:24]
        if best <= 4:
            break
    return best, tuple(memory)


@lru_cache(maxsize=1)
def _c84_bank(G):
    """Pure deterministic bank of fresh asymmetric construction tasks."""
    from math import gcd
    assert G.q == 84
    assert A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    q, mask = G.q, (1 << G.q)-1
    za = [[[], []]]
    zb = [[[], [], []]]

    # z=(1,2) has order 42. Group operations constrain supports only;
    # every parity-matrix action is extracted from a verified unit lift.
    powers, value = [], G.e
    for _ in range(42):
        powers.append(value)
        value = G.mul(value, (1, 2))
    assert value == G.e and len(set(powers)) == 42
    gaps = {d: [t for t in range(1, 42) if gcd(t, 42) == d]
            for d in (1, 2, 3)}

    atlas = {}
    for side, count in (("A", 2), ("B", 3)):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(count)]]
            unit[0][0] = [g]
            A, B = (unit, zb) if side == "A" else (za, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            rows = ar if side == "A" else br
            blocks.append(tuple(_r2_columns(rows, count*q)[:q]))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*(len(grid[0])*q)
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for h, contribution in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+h] ^= contribution
        return tuple(cols)

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        recipes = _C84_CARDS[architecture]
        local = random.Random(
            BASE_SEED*100003 + 84031981 + 104729*slot)

        def draw(side):
            recipe = recipes[0 if side == "A" else 1]
            grid, used = [], set()
            for j, divisor in enumerate(recipe):
                best = None
                for _ in range(48):
                    anchor = G.pick(local)
                    second = G.mul(anchor, local.choice(
                        [powers[t] for t in gaps[divisor]]))
                    opposite = [g for g in G.elements
                                if (g[1]-anchor[1]) % 2 == 1]
                    if mode == 1:
                        wanted = (anchor[1]+1+2*(j % 2)) % 4
                        opposite = [g for g in opposite if g[1] == wanted]
                    elif mode == 2:
                        wanted = (anchor[0]+second[0]+j+architecture) % 21
                        opposite = [g for g in opposite if g[0] == wanted]
                    elif mode == 3:
                        wanted = (anchor[1]+3) % 4
                        opposite = [g for g in opposite
                                    if g[1] == wanted
                                    and g[0] != anchor[0]]
                    third = local.choice(opposite)
                    cell = [anchor, second, third]
                    relatives = (
                        {G.mul(g, G.inv(h)) for g in cell for h in cell
                         if g != h} if side == "A" else
                        {G.mul(G.inv(g), h) for g in cell for h in cell
                         if g != h})
                    penalty = 6-len(relatives)+2*len(relatives & used)
                    candidate = (penalty, cell, relatives)
                    if best is None or penalty < best[0]:
                        best = candidate
                    if penalty == 0:
                        break
                _, cell, relatives = best
                assert len(set(cell)) == 3
                used.update(relatives)
                grid.append(cell)
            return [grid]

        def pool(side, batch):
            width = (2 if side == "A" else 3)*q
            strata, seen = {}, set()

            def key(item):
                return (item["d"], -item["cycles"])

            def admit(grid):
                labels = tuple(tuple(sorted(cell)) for cell in grid[0])
                if labels in seen:
                    return
                seen.add(labels)
                cols = compile_component(side, grid)
                if not _r2_short(cols):
                    return
                rows = _c84_rows(cols, q)
                words = []
                d = _r2_probe(cols, words, local, 3)
                if not words or d > width:
                    return
                item = dict(grid=grid, cols=cols, words=words,
                            d=d, rank=gf2_rank(rows),
                            cycles=_four_cycles(rows))
                bucket = strata.setdefault(item["rank"], [])
                bucket.append(item)
                bucket.sort(key=key, reverse=True)
                del bucket[3:]

            quota = (72 if side == "A" else 112)+32*batch
            for _ in range(quota):
                admit(draw(side))
            for step in range(24 if side == "A" else 40):
                candidates = [x for bucket in strata.values() for x in bucket]
                if not candidates:
                    admit(draw(side))
                    continue
                candidates.sort(key=key, reverse=True)
                parent = local.choice(candidates[:6])
                grid = [[list(cell) for cell in parent["grid"][0]]]
                pressure = [
                    1+sum(((w >> (j*q)) & mask).bit_count()
                          for w in parent["words"][:8])
                    for j in range(width//q)]
                j = local.choices(range(width//q),
                                  weights=pressure, k=1)[0]
                grid[0][j] = draw(side)[0][j]
                admit(grid)

            candidates = [x for bucket in strata.values() for x in bucket]
            candidates.sort(key=key, reverse=True)
            candidates = candidates[:8]
            for item in candidates:
                item["d"] = min(item["d"], _r2_probe(
                    item["cols"], item["words"], local, 24))
            candidates.sort(key=key, reverse=True)
            candidates = candidates[:4]
            for item in candidates:
                item["d"] = min(item["d"], _r2_probe(
                    item["cols"], item["words"], local, 48))
            return sorted(candidates, key=key, reverse=True)

        def merit(item):
            d = min(item["a"]["d"], item["b"]["d"],
                    item["dx"], item["dz"])
            return (item["k"]*d*d/item["n"], d,
                    item["dx"]+item["dz"], -item["cycles"])

        chosen = None
        for batch in range(2):
            ap, bp = pool("A", batch), pool("B", batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not 275*n <= 1000*k < 325*n:
                        continue
                    ar, br = _component_checks(hx, hz, q)
                    assert tuple(_r2_columns(ar, 2*q)) == a["cols"]
                    assert tuple(_r2_columns(br, 3*q)) == b["cols"]
                    d = min(a["d"], b["d"])
                    proposals.append((k*d*d, a, b, hx, hz, n, k))
            proposals.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for _, a, b, hx, hz, n, k in proposals:
                if _c84_low4(_r2_columns(hx, n), hz):
                    continue
                if _c84_low4(_r2_columns(hz, n), hx):
                    continue
                fp = _r2_signature(hx, hz, n, k)
                if fp in emitted:
                    continue
                az = [
                    sum(((w >> (j*q)) & mask) << ((3*j+beta)*q)
                        for j in range(2))
                    for w in a["words"][:16] for beta in range(3)]
                bx = [w << (j*3*q)
                      for w in b["words"][:16] for j in range(2)]
                assert all(_c84_syndrome(w, _r2_columns(hx, n)) == 0
                           for w in az)
                assert all(_c84_syndrome(w, _r2_columns(hz, n)) == 0
                           for w in bx)
                dx, xw = _c84_full(hz, hx, n, local, 6, bx)
                dz, zw = _c84_full(hx, hz, n, local, 6, az)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    a=a, b=b, hx=hx, hz=hz, n=n, k=k, fp=fp,
                    dx=dx, dz=dz, xw=xw, zw=zw,
                    cycles=a["cycles"]+b["cycles"]))
                if len(finalists) == 3:
                    break

            # Equal independent full-code budgets for every finalist.
            for item in finalists:
                dx, _ = _c84_full(
                    item["hz"], item["hx"], item["n"], local, 18,
                    item["xw"])
                dz, _ = _c84_full(
                    item["hx"], item["hz"], item["n"], local, 18,
                    item["zw"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
            if not finalists:
                continue
            winner = max(finalists, key=merit)
            A, B = winner["a"]["grid"], winner["b"]["grid"]
            assert css_ok(winner["hx"], winner["hz"])
            assert winner["n"] == 588 and winner["k"] >= 168
            assert max(row.bit_count()
                       for row in winner["hx"]+winner["hz"]) == 12
            assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
            assert _no_dead_lines([[3, 3]], [[3, 3, 3]])
            emitted.add(winner["fp"])
            chosen = tuple(
                tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for grid in (A, B))
            break
        # Failed tasks never call an archived or previously emitted bank.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the fresh asymmetric cubic task, with every cell defined."""
    item = _c84_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh asymmetric cubic filling passed screening")
    A, B = item
    return {"A": [[list(c) for c in row] for row in A],
            "B": [[list(c) for c in row] for row in B]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly selected cubic cell without new draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: The Dic11 parent's lightest logicals are always sector-one-only; measured slot 0 contained two weight-24 X logicals touching both sectors, including Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Noncommuting quartics, central order-three pair exclusion and the original C13-fiber/component probes suppress weight-nine/eighteen witnesses on C13:C9; current-parent slots 5,6,25,26,45,46 read d2s=9, numerous others read eighteen, and measured slot 28 reaches d30s=22.
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
