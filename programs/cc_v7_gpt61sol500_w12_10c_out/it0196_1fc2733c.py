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
CONCEPT = ("C7 semidirect C12 with action 2, order 84, shapes (1,2)x(1,3): noncommuting trinomials selected against ordinary and order-three-invariant component witnesses should produce five measured d30s readings of at least twelve in cell 0.300; this is a falsifiable search hypothesis.")
GROUP = ("Jump from the order-117 C13:C9 group to C7:C12, y*x*y^-1=x^2. Since 2^3=1 modulo 7, the action defines a valid order-84 group with centre C4. This centre-four order-84 bed is absent from BEDS. Other order-84 groups reached fitness 38.0 at these shapes, and the related centre-four order-80 bed reached 27.5; neither measurement establishes distance transfer.")
SHAPES = ("Target cell 0.300 with A=(1,2), B=(1,3): n=588, kb=168 and guaranteed rate 2/7. A has 168 variables and 84 checks; B has 252 variables and 84 checks. These shapes allow q<=85 and the fixed group uses q=84. Parent measured sector-one minima concentrated on Acol1 or Bcol1 across the opposite columns. Optimize the longer B component separately and require 0.275<=k/n<0.325 using exact assembled ranks.")
FILLING = ("Twenty cards cross five quotient-C12 occupancy arrangements with four rotation-residue policies. All five cells contain three distinct elements, giving HX weight 6+3=9 and HZ weight 9+3=12. Reject commuting triples and central order-two/four pair relatives. Compile component contributions only from verified unit-cell lifts, retain normal-C7-fiber and commuting order-three-orbit witnesses, and independently deepen ordinary searches. Actual stabilizer-dressed embeddings, full-matrix kernel exclusion through four, exact ranks and unequal uncoloured Tanner invariants select fresh designs. Restricted screens and witness readings are not quantum-distance certificates.")
FAMILY_NAME = "r030_C7semiC12_12x13_triple_norm3_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # x^7=y^12=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    # The central subgroup <y^3> is C4.
    if not valid_q(q):
        raise ValueError("C7:C12 requires order 84")
    els = [(a, b) for b in range(12) for a in range(7)]
    action = [pow(2, b, 7) for b in range(12)]

    def red(g):
        return (g[0] % 7, g[1] % 12)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 7, (b+d) % 12)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 12
        return ((-action[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b + a

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
    (0, 1, 2),
    (0, 1, 4),
    (0, 0, 1),
    (0, 1, 5),
    (0, 1, 6),
    (0, 1, 7),
    (0, 2, 5),
    (0, 0, 2),
)

# Acol0,Acol1,Bcol0,Bcol1,Bcol2; four residue policies per card.
# Every slot optimizes fresh relative voltages, not a transported filling.
_R2_CARDS = (
    (0, 0, 0, 0, 0),
    (0, 1, 2, 3, 4),
    (2, 7, 0, 5, 6),
    (1, 3, 2, 4, 7),
    (4, 6, 1, 3, 5),
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
    """True kernel words restricted to normal-C7 fiber indicators."""
    q = G.q
    quotient = sorted({g[1] for g in G.elements})
    fibers = [
        sum(1 << G.idx(g) for g in G.elements if g[1] == b)
        for b in quotient]
    expanded = [fiber << (j*q)
                for j in range(len(cols)//q) for fiber in fibers]
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
        if best <= 4:
            break
    seeds[:] = memory
    return best


def _r2_dressed(item, q):
    """Validate asymmetric embeddings and shorten actual logical classes."""
    hx, hz = item["hx"], item["hz"]
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q)-1
    best = item["n"]+1
    ceiling = min(item["da"], item["db"])+6
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        span = _r2_basis(stabilizers)
        for seed in sorted(set(seeds), key=int.bit_count)[:10]:
            if seed.bit_count() > ceiling:
                continue
            for fixed in range(cb if side == "A" else ca):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((cb*j+fixed)*q) for j in range(ca))
                else:
                    word = seed << (fixed*cb*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    raise ValueError("verified component embedding failed")
                if not _r2_outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(4):
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
def _r3_atlas(G):
    """F2-linear contributions compiled exclusively from verified lifts."""
    q = G.q
    mask = (1 << q)-1
    stub_a = [[[G.e] for _ in range(A_SHAPE[1])]]
    stub_b = [[[G.e] for _ in range(B_SHAPE[1])]]
    tables, partitions = {}, {}
    generator = (0, 4)  # Order three, not the central C4 generator.

    for side, width in (("A", A_SHAPE[1]), ("B", B_SHAPE[1])):
        blocks = []
        for g in G.elements:
            unit = [[[g]] + [[] for _ in range(width-1)]]
            A, B = (unit, stub_b) if side == "A" else (stub_a, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            rows = ar if side == "A" else br
            blocks.append(tuple(_r2_columns(
                [row & mask for row in rows], q)))
        tables[side] = tuple(blocks)

        # A's left action commutes with right translation; B's right
        # action commutes with left translation. These are witness
        # partitions, not a hand-written implementation of either lift.
        covered, orbits = set(), []
        for g in G.elements:
            if G.idx(g) in covered:
                continue
            h, orbit = g, []
            for _ in range(3):
                orbit.append(G.idx(h))
                h = (G.mul(h, generator) if side == "A"
                     else G.mul(generator, h))
            assert h == g and len(set(orbit)) == 3
            covered.update(orbit)
            orbits.append(tuple(orbit))
        assert len(covered) == q
        partitions[side] = tuple(orbits)
    return tables, partitions


def _r3_cell(rng, G, pattern, mode):
    """Noncommuting relative triples with four distinct residue policies."""
    for _ in range(256):
        phase = rng.randrange(12)
        direction = rng.choice((1, 5, 7, 11))
        bs = [(phase+direction*b) % 12 for b in _R2_PATTERNS[pattern]]
        if mode == 0:
            residues = rng.sample(range(7), 3)
        elif mode == 1:
            base, step = rng.randrange(7), rng.randrange(1, 7)
            residues = [(base+t*step) % 7 for t in range(3)]
        elif mode == 2:
            residues = [rng.randrange(7) for _ in range(3)]
        else:
            base, u = rng.randrange(7), rng.randrange(1, 7)
            legal = [v for v in range(1, 7)
                     if v != u and v != 2*u % 7
                     and u != 2*v % 7 and (u+v) % 7]
            v = rng.choice(legal)
            residues = [base, (base+u) % 7, (base+v) % 7]
        cell = list(zip(residues, bs))
        if len(set(cell)) != 3:
            continue
        a = G.mul(G.inv(cell[0]), cell[1])
        b = G.mul(G.inv(cell[0]), cell[2])
        if G.mul(a, b) == G.mul(b, a):
            continue
        # Central nonidentity relatives have order two or four.
        if any(relative[0] == 0 and relative[1] in (3, 6, 9)
               for i, g in enumerate(cell) for h in cell[:i]
               for relative in (G.mul(G.inv(g), h),)):
            continue
        return cell
    raise ValueError("noncommuting trinomial recipe exhausted")


def _r3_pool(rng, G, side, policies, mode, quota):
    """Separately optimize each component, including its orbit subspace."""
    q = G.q
    tables, partitions = _r3_atlas(G)
    atlas, orbits = tables[side], partitions[side]
    raw, seen = [], set()

    for attempt in range(quota):
        cells = [[_r3_cell(rng, G, p, mode) for p in policies]]
        label = tuple(tuple(sorted(c)) for c in cells[0])
        if label in seen:
            continue
        seen.add(label)
        cols = [0]*(len(policies)*q)
        for j, cell in enumerate(cells[0]):
            for g in cell:
                for h, syndrome in enumerate(atlas[G.idx(g)]):
                    cols[j*q+h] ^= syndrome
        rank = gf2_rank(cols)
        if rank < q-8 or not _r2_short(cols):
            continue

        folded, masks = [], []
        for j in range(len(policies)):
            for orbit in orbits:
                syndrome, word = 0, 0
                for h in orbit:
                    syndrome ^= cols[j*q+h]
                    word |= 1 << (j*q+h)
                folded.append(syndrome)
                masks.append(word)
        rows = _r2_columns(cols, q)
        cycles = _four_cycles(rows)
        clean = _r2_short(folded)
        item = dict(cells=cells, cols=cols, folded=folded,
                    masks=masks, rank=rank, cycles=cycles,
                    clean=clean, attempt=attempt)
        raw.append(item)
        raw.sort(key=lambda x: (
            x["clean"], -x["cycles"], x["rank"], -x["attempt"]),
            reverse=True)
        del raw[10:]

    def expand(word, masks):
        full = 0
        while word:
            bit = word & -word
            full ^= masks[bit.bit_length()-1]
            word ^= bit
        return full

    def rescreen(item, ordinary_trials, folded_trials):
        fd = _r2_probe(item["folded"], item["fw"], rng, folded_trials)
        additions = [expand(w, item["masks"]) for w in item["fw"]]
        item["words"][:] = sorted(
            set(item["words"]+additions), key=int.bit_count)[:48]
        od = _r2_probe(item["cols"], item["words"], rng, ordinary_trials)
        item["d"] = min(item["d"], od, 3*fd)

    def merit(item):
        load = sum(w.bit_count() <= item["d"]+1
                   for w in item["words"])
        return (item["d"], -load, item["clean"],
                -item["cycles"], item["rank"])

    for item in raw:
        item["fw"] = []
        item["words"] = _r2_fiber_seeds(item["cols"], G)
        item["d"] = len(item["cols"])+1
        rescreen(item, 6, 8)
    raw.sort(key=merit, reverse=True)
    raw = raw[:4]
    for item in raw:
        rescreen(item, 24, 24)
    raw.sort(key=merit, reverse=True)
    raw = raw[:3]
    for item in raw:
        # Equal independent budgets; discoveries can only lower readings.
        rescreen(item, 48, 48)
    raw.sort(key=merit, reverse=True)
    return raw


@lru_cache(maxsize=1)
def _r2_bank(G):
    """Twenty structural tasks; emit only invariant-distinct fresh codes."""
    assert G.q == 84
    q = G.q
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    bank, emitted = [], set()

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("logical", item["n"]+1))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    for slot in range(N_SLOTS):
        card, mode = divmod(slot % 20, 4)
        policy = _R2_CARDS[card]
        rng = random.Random(BASE_SEED*10007 + 84030073 + 104729*slot)
        chosen = None

        for retry in range(3):
            quota = 64 + 24*retry + 8*(slot//20)
            ap = _r3_pool(rng, G, "A", policy[:2], mode, quota)
            bp = _r3_pool(rng, G, "B", policy[2:], mode, quota)
            candidates = []
            for a in ap:
                for b in bp:
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 588 or not (275*n <= 1000*k < 325*n):
                        continue
                    item = dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        da=a["d"], db=b["d"],
                        aw=list(a["words"]), bw=list(b["words"]),
                        cycles=a["cycles"]+b["cycles"])
                    candidates.append(item)
            candidates.sort(key=merit, reverse=True)

            screened = []
            for item in candidates[:4]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                # Deliberately conservative: exclude all commuting
                # supports through four, including small stabilizers.
                if not _r2_short(_r2_columns(hx, n)):
                    continue
                if not _r2_short(_r2_columns(hz, n)):
                    continue
                item["logical"] = _r2_dressed(item, q)
                if item["logical"] > n:
                    continue
                screened.append(item)

            for item in sorted(screened, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                signature = _r2_signature(hx, hz, n, item["k"])
                if signature in emitted:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx) == 9
                assert max(row.bit_count() for row in hz) == 12
                assert all(len(c) == len(set(c)) == 3
                           for grid in (item["A"], item["B"])
                           for row in grid for c in row)
                emitted.add(signature)
                chosen = (item["A"], item["B"])
                break
            if chosen is not None:
                break

        # A failed task voids only its slot; never substitute a prior
        # filling, group-automorphism image, or block-permutation image.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this slot's independently optimized asymmetric triple design."""
    item = _r2_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh trinomial design passed target-rate screening")
    A, B = item
    return {"A": [[list(c) for c in row] for row in A],
            "B": [[list(c) for c in row] for row in B]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Short rank-weighted component rescreening suffices to remove the sector-1 weight-24 bottleneck; current-parent measured slots 2 and 23 still have d30s=24 with k=52 and 51, while slots 11,22,32 have k=44 and d30s=26.
# REFUTED: The Dic11 parent's lightest logicals are always sector-one-only; measured slot 0 contained two weight-24 X logicals touching both sectors, including Arow0/Brow0 or Arow0/Brow1.
# REFUTED: Noncommuting quartic recipes and central-order-three pair exclusion prevent weight-nine logical witnesses on C13:C9; parent slots 5,6,25,26,45,46 instead returned d2s=9.
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
