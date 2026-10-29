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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): enumerated C4-fixed component witnesses and independently repaired component reservoirs should raise five-code fitness above 53.4240 while retaining k/n<0.125.")
GROUP = ("Frozen L1 Dic11 presentation and order. RUN STATE records 54.9 on this bed and 60.3 on the same-shape cyclic order-46 bed. Dic11's order-four subgroup supplies a structured witness space beyond the parent's normal-C11 fiber screen; all ranks and syndromes use verified lifts.")
SHAPES = ("Frozen equal (2,3) sides target cell 0.100: n=572, guaranteed k>=44 and rate>=1/13, with two 88-check/132-variable components. The size window allows q<=46; q=44 stays fixed. Parent slot 6 has weight-24 sector-1 Z minima across B-column blocks, and slot 48 has weight-24 X minima concentrated in Acol1/Acol2. All measured parent minima are in sector 1. Exact assembled ranks enforce k/n<0.125.")
FILLING = ("Preserve slot-specific degree geometries and add independently affine quadratic-residue C11 cell coordinates. Enumerate kernels on order-four coset indicators when their dimension is at most sixteen, retaining actual witnesses alongside C11 fiber words. Optimize and repeatedly rescreen separate rank-diverse A/B reservoirs, repair witness-heavy columns, and evaluate their combinations using exact assembled ranks. Verified non-stabilizer embeddings and single/pair stabilizer dressings can lower final selection readings. Restricted and randomized witness searches are not quantum-distance certificates.")
FAMILY_NAME = "r010_dic11_23x23_C4norm_componentreservoir_QR_v5"

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
    """Optimize one support geometry with rank-aware witness rescreening."""
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
        if model == 4:
            # Affine copies are chosen independently in each cell, not
            # by one common group automorphism of a completed filling.
            residues = (1, 3, 4, 5, 9)
            recipes = {
                1: ((0,),),
                2: ((0, 1), (0, 2), (0, 3)),
                3: ((0, 1, 2), (0, 0, 1), (0, 1, 3)),
                4: ((0, 0, 1, 1), (0, 0, 1, 2), (0, 1, 2, 3)),
                5: ((0, 0, 1, 2, 3),),
            }
            grid = []
            for i in range(2):
                row = []
                for j in range(3):
                    count = counts[3*i+j]
                    offsets = recipes[count][
                        (idx+2*i+j) % len(recipes[count])]
                    for restart in range(64):
                        phase = rng.randrange(4)
                        direction = rng.choice((-1, 1))
                        u, v = rng.randrange(1, 11), rng.randrange(11)
                        coords = rng.sample(residues, count)
                        cell = []
                        for t, offset in zip(coords, offsets):
                            c = (phase+direction*offset) % 4
                            b, parity = c % 2, c // 2
                            g = (2*((u*t+v) % 11)+parity, b)
                            if any(G.mul(G.inv(h), g) == involution
                                   for h in cell):
                                break
                            cell.append(g)
                        if len(cell) == count:
                            break
                    else:
                        cell = _label_grid(rng, G, counts)[i][j]
                    assert len(cell) == len(set(cell)) == count
                    row.append(cell)
                grid.append(row)
            return grid

        if model == 0:
            grid = _label_grid(rng, G, counts)
            for i in range(2):
                for j in range(3):
                    if counts[3*i+j] != 2:
                        continue
                    # Same C4 fiber means a nonidentity order-11 ratio.
                    # Other legal doubletons have relative order 4 or 22.
                    while True:
                        cell = grid[i][j]
                        relative = G.mul(G.inv(cell[0]), cell[1])
                        if relative[1] or relative[0] % 2:
                            break
                        grid[i][j] = _label_grid(rng, G, counts)[i][j]
            return grid

        long_pairs = set()
        if model == 3:
            from itertools import combinations
            positions = [(i, j) for i in range(2) for j in range(3)
                         if counts[3*i+j] == 2]
            matchings = []
            for size in range(3):
                for subset in combinations(positions, size):
                    if (len({i for i, j in subset}) == size
                            and len({j for i, j in subset}) == size):
                        matchings.append(subset)
            largest = max(map(len, matchings))
            matchings = [m for m in matchings if len(m) == largest]
            long_pairs = set(
                matchings[(idx + sum(counts)) % len(matchings)])

        grid = []
        for i in range(2):
            row = []
            for j in range(3):
                count = counts[3*i+j]
                phase = rng.randrange(4)
                if model == 1:
                    direction = rng.choice((-1, 1))
                    quota = [(phase + direction*t) % 4
                             for t in range(count)]
                elif model == 3:
                    if count == 1:
                        quota = [phase]
                    elif count == 2:
                        # Quotient difference 2: same y coset and odd
                        # rotation difference. Excluding x^11 leaves
                        # relative order 22. Odd differences give order 4.
                        delta = (2 if (i, j) in long_pairs
                                 else rng.choice((1, 3)))
                        quota = [phase, (phase+delta) % 4]
                    elif count == 3:
                        omitted = rng.randrange(4)
                        quota = [(phase+t) % 4 for t in range(4)
                                 if t != omitted]
                    else:
                        quota = [(phase+t) % 4 for t in range(count)]
                elif count == 1:
                    quota = [phase]
                elif count == 2:
                    quota = [phase, (phase + rng.choice((1, 2, 3))) % 4]
                elif count == 3:
                    quota = [phase, phase,
                             (phase + rng.choice((1, 3))) % 4]
                else:
                    quota = [phase, phase, (phase+1) % 4, (phase+2) % 4]
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

    # A's left action commutes with right H translations; B's right
    # action commutes with left H translations. These construct supports
    # only. Every restricted syndrome is evaluated on verified columns.
    subgroup = ((0, 0), (0, 1), (11, 0), (11, 1))
    norm_atoms = {}
    for side in ("A", "B"):
        unseen, partition = set(range(q)), []
        while unseen:
            g = G.elements[min(unseen)]
            members = {
                G.idx(G.mul(g, h) if side == "A" else G.mul(h, g))
                for h in subgroup}
            assert len(members) == 4
            unseen.difference_update(members)
            partition.append(sum(1 << c for c in members))
        assert len(partition) == 11
        norm_atoms[side] = [
            word << (j*q) for j in range(3) for word in partition]

    def c4_words(cols, side):
        """Exact kernel on 33 disjoint C4-orbit indicators.

        Enumeration is complete only within this restricted subspace.
        Candidates with restricted dimension above sixteen are skipped,
        rather than interpreting a truncated enumeration as exhaustive.
        """
        atoms = norm_atoms[side]
        pivots, deps = {}, []
        for c, atom in enumerate(atoms):
            syndrome, remaining = 0, atom
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            word = 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                deps.append(word)
        if len(deps) > 16:
            return None
        space = [0]
        for dep in deps:
            space += [word ^ dep for word in space]
        shortest = sorted(space[1:],
                          key=lambda v: (v.bit_count(), v))[:40]
        expanded = []
        for word in shortest:
            full = 0
            while word:
                bit = word & -word
                full ^= atoms[bit.bit_length()-1]
                word ^= bit
            expanded.append(full)
        return expanded

    def probe(cols, structured, trials):
        """Retain true kernel supports across fresh information sets.

        C4-fixed seeds, C11-fixed seeds, basis words and shortened XORs
        are all actual nonzero witnesses. Readings are upper estimates.
        """
        structured[:] = sorted(set(structured),
                               key=lambda v: (v.bit_count(), v))[:40]
        best = min((v.bit_count() for v in structured), default=width+1)
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            discoveries = list(structured)
            pivots, basis = {}, []
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
                    basis.append(word)
                    best = min(best, word.bit_count())

            for sweep in range(2 + int(trials >= 24)):
                rng.shuffle(basis)
                changed = False
                for a in range(len(basis)):
                    word, weight = basis[a], basis[a].bit_count()
                    for b, other in enumerate(basis):
                        if a == b:
                            continue
                        combined = word ^ other
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                            changed = True
                    basis[a] = word
                    best = min(best, weight)
                if not changed:
                    break

            discoveries.extend(basis)
            words = sorted(set(basis + structured),
                           key=lambda v: (v.bit_count(), v))
            seeds = list(words[:8])
            for a, word in enumerate(words):
                best = min(best, word.bit_count())
                for other in words[:a]:
                    combined = word ^ other
                    if not combined:
                        continue
                    cw = combined.bit_count()
                    best = min(best, cw)
                    if cw <= best+4:
                        discoveries.append(combined)
                    if len(seeds) < 16:
                        seeds.append(combined)
                    elif cw < seeds[-1].bit_count():
                        seeds[-1] = combined
                        seeds.sort(key=int.bit_count)

            rng.shuffle(words)
            for word in sorted(set(seeds), key=int.bit_count)[:8]:
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in words:
                        combined = word ^ other
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                            changed = True
                    if not changed:
                        break
                best = min(best, weight)
                discoveries.append(word)
            structured[:] = sorted(
                set(discoveries),
                key=lambda v: (v.bit_count(), v))[:32]
            if best <= 20:
                break
        return best

    def component_key(item):
        d = min(item["d"], 34)
        load = sum(word.bit_count() <= d+1 for word in item["words"])
        return (d, -load, -item["cycles"], item["rank"])

    def trim(beam, limit=12):
        # Do not erase all rank-deficient alternatives before measuring
        # their actual assembled LP rank. Component rank is not LP k.
        ranks, kept = {}, []
        for item in sorted(beam, key=component_key, reverse=True):
            rank = item["rank"]
            if ranks.get(rank, 0) >= 3:
                continue
            kept.append(item)
            ranks[rank] = ranks.get(rank, 0)+1
            if len(kept) == limit:
                break
        beam[:] = kept

    zero = [[[] for _ in range(3)] for _ in range(2)]
    reservoirs = {}
    mask = (1 << q)-1

    for side, counts in (("A", ta), ("B", tb)):
        beam, seen_component = [], set()

        def admit_component(grid):
            signature = tuple(tuple(sorted(cell))
                              for row in grid for cell in row)
            if signature in seen_component:
                return
            seen_component.add(signature)
            A, B = (grid, zero) if side == "A" else (zero, grid)
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_checks(hx, hz, q)
            rows = ar if side == "A" else br
            cols = columns(rows)
            norms = c4_words(cols, side)
            if norms is None:
                return
            words = constant_fiber_words(cols) + norms
            item = {
                "grid": grid, "cols": cols, "words": words,
                "rank": gf2_rank(rows), "cycles": _four_cycles(rows),
                "d": probe(cols, words, 4),
            }
            beam.append(item)
            trim(beam)

        def refine_component(trials):
            for item in beam:
                item["d"] = min(
                    item["d"],
                    probe(item["cols"], item["words"], trials))
            trim(beam)

        for attempt in range(96):
            admit_component(draw_grid(
                counts, (idx+attempt//16) % 5))
        refine_component(12)

        for step in range(48):
            if not beam:
                admit_component(draw_grid(counts, (idx+step) % 5))
                continue
            parent = beam[rng.randrange(min(5, len(beam)))]
            grid = [[list(cell) for cell in row]
                    for row in parent["grid"]]
            proposal = draw_grid(counts, (idx+step) % 5)
            pressure = [
                1 + sum(((word >> (j*q)) & mask).bit_count()
                        for word in parent["words"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = proposal[i][j]
            if step % 6 == 5:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = proposal[1-i][other]
            admit_component(grid)
            if step % 12 == 11:
                # Repairs compete against periodically rescreened
                # readings, not indefinitely lucky shallow estimates.
                refine_component(8)

        if not beam:
            raise ValueError("empty C4-screened component reservoir")
        refine_component(24)
        trim(beam, 6)
        refine_component(64)
        reservoirs[side] = list(beam)

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("logical", width+1), 32)
        load = sum(word.bit_count() <= d+1
                   for word in item["aw"]+item["bw"])
        return (item["k"]*d*d, d, -load,
                min(item["da"], 36)+min(item["db"], 36),
                -item["cycles"])

    finalists = []
    for ai in reservoirs["A"]:
        for bi in reservoirs["B"]:
            A, B = ai["grid"], bi["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (q <= k and 8*k < n):
                continue
            finalists.append({
                "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
                "ac": ai["cols"], "bc": bi["cols"],
                "aw": list(ai["words"]), "bw": list(bi["words"]),
                "da": ai["d"], "db": bi["d"],
                "cycles": ai["cycles"]+bi["cycles"],
            })

    if not finalists:
        raise ValueError("no reservoir pairing in target rate cell")
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:8]

    def logical_reading(item):
        """Verify embeddings and logical classes before stabilizer dressing."""
        best = item["n"]+1

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

        for side, words, checks, stabilizers in (
                ("A", item["aw"], item["hx"], item["hz"]),
                ("B", item["bw"], item["hz"], item["hx"])):
            pivots = row_basis(stabilizers)
            seeds = []
            for seed in sorted(set(words), key=int.bit_count)[:8]:
                for fixed in range(3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*3*q)
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if outside(word, pivots):
                        seeds.append(word)
                        best = min(best, word.bit_count())

            for word in sorted(set(seeds), key=int.bit_count)[:8]:
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        combined = word ^ row
                        cw = combined.bit_count()
                        nearby.append((cw, row))
                        if cw < weight:
                            chosen, weight = combined, cw
                    nearby.sort(key=lambda pair: pair[0])
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

    # Pair-specific fresh searches can discover witnesses not retained
    # when selecting either independent component reservoir.
    for item in finalists:
        item["da"] = min(
            item["da"], probe(item["ac"], item["aw"], 24))
        item["db"] = min(
            item["db"], probe(item["bc"], item["bw"], 24))
        item["logical"] = logical_reading(item)

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert q <= k and 8*k < n
    assert _grid_weights(
        [[len(cell) for cell in row] for row in A],
        [[len(cell) for cell in row] for row in B]) <= 12
    assert all(len(cell) == len(set(cell)) <= 5
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the adaptively optimized full cell without subsequent redraw."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rank surplus alone predicts stronger distance readings: parent slot 0 had k=60 and d2s=22, while slots 11/22/32 had k=44 and d30s=26.
# REFUTED: Matching-controlled order-22 binomials and independently deepened scalar component probes eliminate the weight-24 sector-1 bottleneck; the measured parent retained d30s=24 at k=56 in slot 6 and k=52 in slot 48.
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
