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
CONCEPT = ("C45, cyclic order 45, shapes (2,3)x(2,3): matched rank-one GF4 quotient reductions, primitive-order doubletons and independently deepened witness screening should produce five fresh cell-0.100 codes exceeding fitness 60.3346.")
GROUP = ("Jump from nonabelian Dic11 to genuinely different cyclic C45, an untried bed in BEDS. Cyclic C46 and C43 reached 60.3 and 54.6 versus this parent's 51.9442. Since 45 is odd, F2[C45] is semisimple; its C3 quotient contains a dimension-two GF4 constituent. Rank one on both 2x3 component reductions contributes eight dimensions above rank counting. Actual assembled ranks, not this prediction, gate every filling.")
SHAPES = ("Keep equal (2,3) sides for cell 0.100: n=585, kb=45, guaranteed rate 1/13, and two equal 135-variable/90-check components. These shapes allow q<=46; this new family fixes q=45. Parent measured weight-24..26 minima occupied sector-one Acol/Bcol blocks, especially Acol1/2 and Bcol1/2. Require verified k>=53 and 8*k<n.")
FILLING = ("Retain canonical degree-distinguished support pairs across augmentation-rank strata. Prescribe residues modulo three so each component's GF4 reduction is an outer product of nonzero row and column factors. Doubletons have primitive order-45 differences; larger cells use distinct mixed-coset occupancy rules and difference packing. Verified lifts supply ranks and component syndromes. Seed commuting cofactors and subgroup-indicator kernels, retain and recombine actual randomized witnesses, repair limiting-component cells, independently deepen finalists and test genuine stabilizer-dressed logicals. Search readings remain heuristics and upper witnesses, never distance certificates.")
FAMILY_NAME = "r010_C45_23x23_GF4rank1_primitive_cofactor_norm_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 45

def custom_ops(q):
    # Additive cyclic C45; projection modulo three gives C3.
    if not valid_q(q):
        raise ValueError("C45 has order 45")
    els = [(a,) for a in range(45)]

    def red(g):
        return (g[0] % 45,)

    def mul(g, h):
        return ((g[0] + h[0]) % 45,)

    def inv(g):
        return ((-g[0]) % 45,)

    def idx(g):
        return g[0] % 45

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
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


def _label_grid(rng, G, counts, model=0, card=0):
    """Construct a rank-one GF4 projection, then optimize actual voltages.

    In GF4, x^0,x^1,x^2 are represented by 1,2,3.
    For odd cell weight the projected support has one odd coset;
    for even weight it has the complementary pair of odd cosets.
    Both evaluate to the prescribed x^(row_factor+column_factor).
    """
    from collections import Counter
    from math import gcd

    assert G.q == 45
    pools = [[g for g in G.elements if g[0] % 3 == t]
             for t in range(3)]
    uf = [rng.randrange(3) for _ in range(2)]
    vf = [rng.randrange(3) for _ in range(3)]
    grid = [[None]*3 for _ in range(2)]
    internal = [Counter(), Counter()]
    cross_used = Counter()
    coefficients = ((3, 2, 3), (5, 4, 1), (2, 3, 5))[model % 3]

    for i in range(2):
        for j in range(3):
            count = counts[3*i+j]
            if not count:
                grid[i][j] = []
                continue
            exponent = (uf[i]+vf[j]) % 3
            a, b = (exponent+1) % 3, (exponent+2) % 3
            if count == 1:
                quota = [exponent]
            elif count == 2:
                quota = [a, b]
            elif count == 3:
                repeated = a if model == 0 else b if model == 1 else rng.choice((a, b))
                quota = [exponent, repeated, repeated]
            elif count == 4:
                repeated = (exponent, a, b)[(card+i+j+model) % 3]
                quota = [a, b, repeated, repeated]
            elif count == 5:
                quota = [exponent, a, a, b, b]
            else:
                raise ValueError("illegal factor-cell count")

            chosen = None
            for attempt in range(16):
                rng.shuffle(quota)
                cell = []
                for coset in quota:
                    available = [g for g in pools[coset] if g not in cell]
                    cell.append(rng.choice(available))
                if count == 2 and gcd((cell[1][0]-cell[0][0]) % 45, 45) != 1:
                    continue

                differences = [
                    (g[0]-h[0]) % 45
                    for g in cell for h in cell if g != h]
                cross = ([] if i == 0 else [
                    (g[0]-h[0]) % 45
                    for g in grid[0][j] for h in cell])
                p, r, c = coefficients
                penalty = (
                    p*(len(differences)-len(set(differences)))
                    + r*sum(internal[i][d] for d in differences)
                    + c*(len(cross)-len(set(cross))
                         + sum(cross_used[d] for d in cross)))
                key = (penalty, rng.random())
                if chosen is None or key < chosen[0]:
                    chosen = (key, cell, differences, cross)

            if chosen is None:
                # The two cosets admit many primitive differences.
                while chosen is None:
                    cell = [rng.choice(pools[a]), rng.choice(pools[b])]
                    if gcd((cell[1][0]-cell[0][0]) % 45, 45) == 1:
                        differences = [(cell[0][0]-cell[1][0]) % 45,
                                       (cell[1][0]-cell[0][0]) % 45]
                        cross = ([] if i == 0 else [
                            (g[0]-h[0]) % 45
                            for g in grid[0][j] for h in cell])
                        chosen = ((0, 0), cell, differences, cross)

            _, cell, differences, cross = chosen
            value = 0
            for g in cell:
                value ^= (1, 2, 3)[g[0] % 3]
            assert value == (1, 2, 3)[exponent]
            assert len(cell) == len(set(cell)) == count
            grid[i][j] = cell
            internal[i].update(differences)
            cross_used.update(cross)
    return grid


def NEW_CTX(rng, G, idx):
    """Degree-distinguished GF4-rank-one task with retained true witnesses."""
    assert G.q == 45
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q, width = G.q, 3*G.q
    mask = (1 << q)-1

    def columns(rows, length):
        out = [0]*length
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def syndrome(cols, word):
        syn = 0
        while word:
            bit = word & -word
            syn ^= cols[bit.bit_length()-1]
            word ^= bit
        return syn

    def dependencies(cols):
        pivots, deps = {}, []
        for c, syn in enumerate(cols):
            word = 1 << c
            while syn:
                p = syn.bit_length()-1
                hit = pivots.get(p)
                if hit is None:
                    pivots[p] = (syn, word)
                    break
                syn ^= hit[0]
                word ^= hit[1]
            if not syn:
                deps.append(word)
        return deps

    def probe(cols, memory, trials):
        """Retain nonzero actual kernels; high readings prove no exclusion."""
        length = len(cols)
        retained = sorted({w for w in memory if w},
                          key=lambda w: (w.bit_count(), w))[:40]
        order = list(range(length))
        best = min((w.bit_count() for w in retained), default=length+1)
        for trial in range(trials):
            rng.shuffle(order)
            pivots, deps = {}, []
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
                    deps.append(word)

            words = sorted(set(retained+deps),
                           key=lambda w: (w.bit_count(), w))
            discoveries = set(words)
            for sweep in range(2):
                reducers = words[:32]
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
                               key=lambda w: (w.bit_count(), w))
                discoveries.update(words)

            short = sorted(discoveries,
                           key=lambda w: (w.bit_count(), w))[:32]
            for a, word in enumerate(short):
                for other in short[:a]:
                    combined = word ^ other
                    if combined:
                        discoveries.add(combined)
            if trials >= 16 and trial % 8 == 0:
                for a, word in enumerate(short[:8]):
                    for b in range(a):
                        for c in range(b):
                            combined = word ^ short[b] ^ short[c]
                            if combined:
                                discoveries.add(combined)
            retained = sorted(discoveries,
                              key=lambda w: (w.bit_count(), w))[:40]
            if retained:
                best = min(best, retained[0].bit_count())
            if best <= 4:
                break
        memory[:] = retained
        return best

    def cofactor_words(grid, cols, inverse=False):
        """Commutative group-algebra minors, syndrome-verified after encoding."""
        entries = [[set(G.inv(g) if inverse else g for g in cell)
                    for cell in row] for row in grid]

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
            minors.append(product(entries[0][s], entries[1][t])
                          ^ product(entries[0][t], entries[1][s]))
        words = []
        for shift in G.elements[::9]:
            word = 0
            for j, minor in enumerate(minors):
                for g in minor:
                    word ^= 1 << (j*q+G.idx(G.mul(g, shift)))
            if word and syndrome(cols, word) == 0:
                words.append(word)
        return words

    def norm_words(cols, size, trials=0):
        """Restrict VERIFIED syndromes to disjoint subgroup indicators.

        This searches a subspace, not unrestricted quantum distance.
        """
        step = q//size
        fibers = [
            sum(1 << G.idx((a+t*step,)) for t in range(size))
            for a in range(step)]
        physical = [fiber << (j*q)
                    for j in range(3) for fiber in fibers]
        folded = [syndrome(cols, word) for word in physical]
        compact = dependencies(folded)
        if size == 15:
            # Nine indicators: enumerate this restricted kernel exactly.
            combinations = [0]
            for dep in compact:
                combinations += [word ^ dep for word in combinations]
            compact = [word for word in combinations if word]
        elif trials:
            probe(folded, compact, trials)

        expanded = []
        for coefficient in compact:
            word = 0
            while coefficient:
                bit = coefficient & -coefficient
                word ^= physical[bit.bit_length()-1]
                coefficient ^= bit
            if word:
                assert syndrome(cols, word) == 0
                expanded.append(word)
        return expanded

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", width+1), 32)
        load = sum(w.bit_count() <= d+1 for w in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                min(item["da"], 36)+min(item["db"], 36),
                -item["cycles"])

    finalists, seen = [], set()

    def admit(A, B):
        labeling = tuple(tuple(sorted(cell)) for grid in (A, B)
                         for row in grid for cell in row)
        if labeling in seen:
            return
        seen.add(labeling)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (q+8 <= k and 8*k < n):
            return
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar, width), columns(br, width)
        aw = cofactor_words(A, ac)+norm_words(ac, 15)
        bw = cofactor_words(B, bc, inverse=True)+norm_words(bc, 15)
        item = dict(
            A=A, B=B, hx=hx, hz=hz, n=n, k=k,
            ac=ac, bc=bc, aw=aw, bw=bw,
            da=probe(ac, aw, 3), db=probe(bc, bw, 3),
            cycles=_four_cycles(ar)+_four_cycles(br))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    for attempt in range(72):
        model = (idx+attempt//24) % 3
        admit(_label_grid(rng, G, ta, model, idx),
              _label_grid(rng, G, tb, model, idx+1))

    # Changing a single residue by a multiple of three preserves its
    # GF4 projection. Repairs alter genuine voltages, not common gauges.
    for step in range(32):
        if not finalists:
            model = (idx+step) % 3
            admit(_label_grid(rng, G, ta, model, idx),
                  _label_grid(rng, G, tb, model, idx+1))
            continue
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        side = (rng.choice(("A", "B")) if step % 5 == 0
                else "A" if parent["da"] <= parent["db"] else "B")
        grid = A if side == "A" else B
        seeds = parent["aw"] if side == "A" else parent["bw"]
        pressure = [
            1+sum(((word >> (j*q)) & mask).bit_count()
                  for word in seeds[:8])
            for j in range(3)]
        j = rng.choices(range(3), weights=pressure, k=1)[0]
        i = rng.randrange(2)
        cell = grid[i][j]
        if not cell:
            continue
        from math import gcd
        for move in range(2 if step % 6 == 5 else 1):
            position = rng.randrange(len(cell))
            residue = cell[position][0] % 3
            available = [
                g for g in G.elements if g[0] % 3 == residue
                and g not in cell]
            rng.shuffle(available)
            for g in available:
                proposal = list(cell)
                proposal[position] = g
                if len(proposal) == 2 and gcd(
                        (proposal[1][0]-proposal[0][0]) % 45, 45) != 1:
                    continue
                cell = proposal
                break
        grid[i][j] = cell
        admit(A, B)

    if not finalists:
        raise ValueError("no C45 GF4-rank-one filling in target cell")

    # Explicitly expose kernels supported on C3, C5 and C9 indicators.
    # These physical witnesses can only LOWER selection readings.
    for item in finalists:
        for cols, words, key in (
                (item["ac"], item["aw"], "da"),
                (item["bc"], item["bw"], "db")):
            for size in (3, 5, 9):
                words.extend(norm_words(cols, size, 12))
            item[key] = min(item[key], probe(cols, words, 20))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]

    def row_basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                hit = pivots.get(p)
                if hit is None:
                    pivots[p] = word
                    break
                word ^= hit
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length()-1
            hit = pivots.get(p)
            if hit is None:
                return True
            word ^= hit
        return False

    def dressed_reading(item):
        """Verify logical classes before single/pair stabilizer dressing."""
        best = item["n"]+1
        for side, seeds, checks, stabilizers in (
                ("A", item["aw"], item["hx"], item["hz"]),
                ("B", item["bw"], item["hz"], item["hx"])):
            span = row_basis(stabilizers)
            logicals = []
            for seed in sorted(set(seeds), key=int.bit_count)[:16]:
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
                    best = min(best, weight)
        return best

    for item in finalists:
        item["da"] = min(item["da"], probe(item["ac"], item["aw"], 80))
        item["db"] = min(item["db"], probe(item["bc"], item["bw"], 80))
        assert all(syndrome(item["ac"], w) == 0 for w in item["aw"])
        assert all(syndrome(item["bc"], w) == 0 for w in item["bw"])
        item["logical"] = dressed_reading(item)

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 585 and q+8 <= k and 8*k < n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert _grid_weights(
        [[len(cell) for cell in row] for row in A],
        [[len(cell) for cell in row] for row in B]) <= 12
    assert all(len(cell) == len(set(cell)) <= 5
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell without further label changes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rank surplus alone predicts stronger distance readings; current-parent slot 0 had k=60 and d2s=22, while measured slots 11,22,32 had k=44 and d30s=26.
# REFUTED: Normal-C11-constant witnesses and scalar component rescreening eliminate the sector-one weight-24 bottleneck; current-parent measured slots 2 and 23 retained d30s=24 at k=52 and 51.
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
