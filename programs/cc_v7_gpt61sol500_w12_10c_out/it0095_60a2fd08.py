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
CONCEPT = ("C13 semidirect C3 with action 3, order 39, A=(1,2), B=(3,5): predominantly degree-four components selected against structured and randomized kernel witnesses should deliver five measured distance readings >=14 in frontier cell 0.150.")
GROUP = ("Jump from Dic11 to the genuinely different odd-order group C13:C3; 3^3=1 modulo 13 validates the action. Its measured (2,3)x(2,3) bed reached fitness 45.2 and distance readings 23-24. The new (1,2)x(3,5) bed is untried; transfer of that performance is a hypothesis.")
SHAPES = ("Target the relatively weak 0.150 frontier: n=13*39=507, kb=2*39=78 and guaranteed rate 2/13. A has 78 variables/39 checks; B has 195 variables/117 checks. Parent minima occupied sector-1 Acol/Bcol blocks, prominently Acol1/Acol2 and Bcol1/Bcol2. Replace both component geometries and require exact 0.125<=k/n<0.175.")
FILLING = ("Enumerate support plans with distinct full Tanner degree multisets, prioritizing four-term A cells and degree-four B columns. Both check-weight sums are explicitly <=12. Search three C3-quotient occupancy models, enumerate normal-C13-constant component kernels, deepen randomized kernel-witness searches independently, and screen finalists for quantum logicals through weight four. Component readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r015_C13sdC3_12x35_degree4_fiber_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # x^13=y^3=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    if not valid_q(q):
        raise ValueError("C13 semidirect C3 requires order 39")
    els = [(a, b) for b in range(3) for a in range(13)]
    action = (1, 3, 9)

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 13, (b+d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


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


def _retired_dic11_ctx(rng, G, idx):
    """Retained parent experiment; never called by the new construction."""
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
                phase = rng.randrange(4)
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

    def probe(cols, structured, trials):
        """Actual component kernel words; readings are not certificates."""
        best = min((v.bit_count() for v in structured), default=width+1)
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = list(structured)
            for c in order:
                syn, word = cols[c], 1 << c
                while syn:
                    p = syn.bit_length()-1
                    if p not in pivots:
                        pivots[p] = (syn, word)
                        break
                    ps, pw = pivots[p]
                    syn ^= ps
                    word ^= pw
                if not syn:
                    words.append(word)
                    best = min(best, word.bit_count())
            words = list(set(words))
            for sweep in range(1 + int(trials > 8)):
                words.sort(key=int.bit_count)
                changed = False
                for a in range(len(words)):
                    word = words[a]
                    weight = word.bit_count()
                    for b in range(a):
                        z = word ^ words[b]
                        zw = z.bit_count()
                        if 0 < zw < weight:
                            word, weight = z, zw
                            changed = True
                    words[a] = word
                    best = min(best, weight)
                if not changed:
                    break
            if best <= 4:
                break
        return best

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    finalists = []
    for attempt in range(72):
        # Each geometry searches all three occupancy models; their order
        # varies by slot, not by a common automorphism of one labeling.
        model = (idx + attempt//24) % 3
        A = draw_grid(ta, model)
        B = draw_grid(tb, model)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (q <= k and 8*k < n):
            continue
        ar, br = _component_checks(hx, hz, q)
        ac, bc = columns(ar), columns(br)
        aw, bw = constant_fiber_words(ac), constant_fiber_words(bc)
        item = {
            "A": A, "B": B, "k": k, "ac": ac, "bc": bc,
            "aw": aw, "bw": bw,
            "da": probe(ac, aw, 2), "db": probe(bc, bw, 2),
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    if not finalists:
        raise ValueError("no positive-k labeling in target rate cell")
    for item in finalists:
        item["da"] = min(item["da"],
                         probe(item["ac"], item["aw"], 8))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 8))
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["da"] = min(item["da"],
                         probe(item["ac"], item["aw"], 32))
        item["db"] = min(item["db"],
                         probe(item["bc"], item["bw"], 32))
    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert q <= k and 8*k < n
    return {"A": A, "B": B}


def _r15_support_plans():
    """Distinct complete Tanner degree multisets, not labeling orbit copies."""
    from itertools import combinations_with_replacement, product
    choices = {
        total: sorted(
            [v for v in product(range(4), repeat=3) if sum(v) == total],
            key=lambda v: (-min(v), max(v), v))
        for total in (3, 4, 5)
    }

    def realize(columns, targets):
        # Construct the count grid by exact row-budget backtracking.
        def visit(j, remaining):
            if j == 5:
                return () if not any(remaining) else None
            left = 4-j
            for column in choices[columns[j]]:
                rest = tuple(remaining[i]-column[i] for i in range(3))
                if min(rest) < 0 or max(rest) > 3*left:
                    continue
                if sum(rest) != sum(columns[j+1:]):
                    continue
                tail = visit(j+1, rest)
                if tail is not None:
                    return (column,) + tail
            return None
        columns_grid = visit(0, targets)
        if columns_grid is None:
            return None
        return tuple(tuple(column[i] for column in columns_grid)
                     for i in range(3))

    proposals = []
    for aw in ((4,4), (3,5), (3,4), (4,5), (3,3),
               (2,5), (2,4), (2,3), (2,2)):
        for bc in combinations_with_replacement((3,4,5), 5):
            if sum(aw)+max(bc) > 12:
                continue
            for br in combinations_with_replacement(
                    range(5, 13-max(aw)), 3):
                if sum(br) != sum(bc):
                    continue
                B = realize(bc, br)
                if B is None:
                    continue
                # HX and HZ check degrees, and both qubit-sector degrees.
                checks = tuple(sorted(
                    [sum(aw)+c for c in bc] +
                    [r+a for r in br for a in aw]))
                qubits = tuple(sorted(
                    [a+c for a in aw for c in bc] +
                    [sum(aw)+r for r in br]))
                zeros = sum(x == 0 for row in B for x in row)
                priority = (min(aw), min(bc), min(br),
                            sum(aw)+sum(br), -max(br), -zeros)
                proposals.append((priority, (checks, qubits), aw, B))
    proposals.sort(reverse=True)
    plans, seen = [], set()
    for _, fingerprint, aw, B in proposals:
        if fingerprint not in seen:
            seen.add(fingerprint)
            plans.append((aw, B))
    if len(plans) < 20:
        raise ValueError("insufficient degree-distinguished rate-0.150 plans")
    return tuple(plans)


_R15_PLANS = _r15_support_plans()


def _r15_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length()-1] |= bit
            row ^= low
    return cols


def _r15_row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _r15_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _r15_small_logical(checks, stabilizers, n):
    """Exact logical existence test through four, including mixed sectors."""
    basis = _r15_row_basis(stabilizers)
    cols = _r15_columns(checks, n)
    seen = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[i]
        elif _r15_outside(bits[i] ^ old, basis):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            s = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif _r15_outside(word ^ old, basis):
                return True
    return False


def _r15_fiber_words(G, cols, blocks):
    """Exact kernel on normal-C13 fiber indicators, using verified syndromes."""
    q = G.q
    fibers = [sum(1 << G.idx(g) for g in G.elements if g[1] == b)
              for b in range(3)]
    indicators = [fiber << (j*q)
                  for j in range(blocks) for fiber in fibers]
    pivots, dependencies = {}, []
    for c, full in enumerate(indicators):
        syndrome = 0
        while full:
            low = full & -full
            syndrome ^= cols[low.bit_length()-1]
            full ^= low
        word = 1 << c
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
    words = [0]
    for dependency in dependencies:
        words += [word ^ dependency for word in words]
    words = sorted(words[1:], key=lambda v: (v.bit_count(), v))[:16]
    result = []
    for word in words:
        full = 0
        while word:
            low = word & -word
            full ^= indicators[low.bit_length()-1]
            word ^= low
        result.append(full)
    return result


def _r15_probe(cols, structured, rng, trials):
    """Actual nonzero classical kernel witnesses; never distance proofs."""
    best = min((v.bit_count() for v in structured),
               default=len(cols)+1)
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, list(structured)
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
                best = min(best, word.bit_count())
        words = list(set(words))
        for sweep in range(2):
            words.sort(key=int.bit_count)
            changed = False
            for a in range(len(words)):
                word, weight = words[a], words[a].bit_count()
                for b in range(a):
                    z = word ^ words[b]
                    zw = z.bit_count()
                    if z:
                        best = min(best, zw)
                    if 0 < zw < weight:
                        word, weight = z, zw
                        changed = True
                words[a] = word
                best = min(best, weight)
            if not changed:
                break
        # Short basis triples supplement the pair-reduction search.
        small = sorted(words, key=int.bit_count)[:10]
        for a, word in enumerate(small):
            for b in range(a):
                pair = word ^ small[b]
                for c in range(b):
                    z = pair ^ small[c]
                    if z:
                        best = min(best, z.bit_count())
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize the slot's degree-distinguished support geometry."""
    q = G.q
    assert q == 39
    aw, sb = _R15_PLANS[idx % len(_R15_PLANS)]
    sa = [list(aw)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    pools = [[g for g in G.elements if g[1] == b] for b in range(3)]

    def draw_cell(count, model):
        if count == 0:
            return []
        phase = rng.randrange(3)
        if count == 1:
            quota = (1,0,0)
        elif count == 2:
            # Same-coset doubletons have relative order 13, not 3.
            quota = (2,0,0) if model != 2 else (1,1,0)
        elif count == 3:
            quota = ((1,1,1), (2,1,0), (3,0,0))[model]
        elif count == 4:
            quota = ((2,1,1), (2,2,0), (3,1,0))[model]
        else:
            quota = ((3,1,1), (2,2,1), (3,2,0))[model]
        direction = rng.choice((-1,1))
        cell = []
        for b, amount in enumerate(quota):
            cell.extend(rng.sample(pools[(phase+direction*b) % 3], amount))
        return cell

    def components(hx, hz):
        # A: fixed B-column zero in HX. B: fixed A-column zero in HZ.
        # All actions and column ordering come from the verified lift.
        mask = (1 << q)-1
        ar = [(row & mask) |
              (((row >> (5*q)) & mask) << q)
              for row in hx[:q]]
        bm = (1 << (5*q))-1
        br = [row & bm for row in hz[:3*q]]
        return ar, br

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    finalists = []
    for attempt in range(48):
        model = (idx+attempt) % 3
        A = [[draw_cell(count, model) for count in aw]]
        B = [[draw_cell(count, (model+i) % 3) for count in row]
             for i, row in enumerate(sb)]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (125*n <= 1000*k < 175*n):
            continue
        ar, br = components(hx, hz)
        ac, bc = _r15_columns(ar, 2*q), _r15_columns(br, 5*q)
        af = _r15_fiber_words(G, ac, 2)
        bf = _r15_fiber_words(G, bc, 5)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": ac, "bc": bc, "af": af, "bf": bf,
            "da": _r15_probe(ac, af, rng, 3),
            "db": _r15_probe(bc, bf, rng, 3),
            "cycles": _four_cycles(ar)+_four_cycles(br),
        }
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[6:]
    if not finalists:
        raise ValueError("no labeling passed the exact 0.150 rate gate")

    for item in finalists:
        item["da"] = min(item["da"],
                         _r15_probe(item["ac"], item["af"], rng, 16))
        item["db"] = min(item["db"],
                         _r15_probe(item["bc"], item["bf"], rng, 16))
    finalists.sort(key=merit, reverse=True)
    for item in finalists[:3]:
        item["da"] = min(item["da"],
                         _r15_probe(item["ac"], item["af"], rng, 48))
        item["db"] = min(item["db"],
                         _r15_probe(item["bc"], item["bf"], rng, 48))
    finalists.sort(key=merit, reverse=True)

    for item in finalists:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        if _r15_small_logical(hx, hz, n):
            continue
        if _r15_small_logical(hz, hx, n):
            continue
        A, B = item["A"], item["B"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert 125*n <= 1000*item["k"] < 175*n
        assert all(len(cell) == len(set(cell))
                   for grid in (A,B) for row in grid for cell in row)
        return {"A": A, "B": B}
    raise ValueError("all finalists contain a weight<=4 logical")


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rank surplus alone guarantees a larger distance reading; parent slots 0/1 had k=60/57 and d2s=22, whereas rank-bound slots 11/22/32 had k=44 and d30s=26.
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
