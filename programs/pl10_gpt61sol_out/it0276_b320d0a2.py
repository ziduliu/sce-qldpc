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
CONCEPT = ("For Dic12, order 48, A=(1,2), B=(2,5), an all-rotation C8-supported unit trinomial paired with a mixed-coset trinomial and joint stabilizer-aware decoder screening should raise cell 0.250's worse-basis fitness above 6.5151.")
GROUP = ("Frozen byte-for-byte under L1. This Dic12 bed measured fitness 6.5151, and a shown alternative filling reached 6.7777. The normal subgroup C8=<x^3> supplies an invertible odd-augmentation trinomial. These facts motivate a controlled filling comparison, not a group-only advantage.")
SHAPES = ("Frozen (1,2)x(2,5): q48 gives n576, kb144 and guaranteed rate 0.250; the shape factor permits q<=49. Components have 96 variables/48 checks and 240 variables/96 checks. Parent failures span both A columns, several B columns and Acol/Brow checks, mostly at weak enrichment. Require complete verified k144 rather than attributing failures to one block or logical index.")
FILLING = ("Twenty contracts cross four B count architectures with five ratio policies. Both A cells have three terms: A0 is the all-rotation unit 1+x^3+x^9, while A1 mixes a rotation of order24/12 with the y coset. B has 15/16 terms, including isolated quartic and paired-zero-trinomial architectures; selected binomials mix cosets. HX weights are <=10, HZ <=11, and six-round CNOT counts are 25056/25920. Helper-derived actions screen rank, connectivity, short kernels and capped sampled protection. Complete-code normalized min-sum probes classify settled residuals modulo actual stabilizers, compare both bases on shared degree-weighted data noise and sampled opposite-check pairs, then validate two finalists with an independent noise book. These probes omit ASC ordering, measurement histories and relay decoding and are not circuit-pL estimates. Target collective basis imbalance: parent slots3/4 both had d2s12 and depth14 but 73/nine X failures per200000. Final matrices enforce k144, CSS and weight legality; prefer no logical<=4.")
FAMILY_NAME = "r025_dic12_q48_12x25_C8unit_jointdecoder_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic12 requires order 48")
    # x^24=1, y^2=x^12, y*x*y^-1=x^-1.
    # Elements are x^a*y^b, with b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(24)]

    def red(g):
        return (int(g[0]) % 24, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        sign = 1 if b == 0 else -1
        return ((a + sign*c + 12*b*d) % 24, (b+d) % 2)

    def inv(g):
        a, b = red(g)
        if b == 0:
            return ((-a) % 24, 0)
        return ((a + 12) % 24, 1)

    def idx(g):
        a, b = red(g)
        return 24*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


def _dic_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _dic_cell(rng, count, palette, majority):
    """Same-coset ratios have prescribed long rotation orders."""
    if count == 0:
        return []
    if count == 1:
        return [(rng.randrange(24), rng.randrange(2))]
    b = rng.randrange(2) if count == 2 else majority % 2
    a = rng.randrange(24)
    cell = [(a, b), ((a + rng.choice(palette)) % 24, b)]
    if count >= 3:
        c = rng.randrange(24)
        cell.append((c, 1-b))
    if count == 4:
        cell.append(((c + rng.choice(palette)) % 24, 1-b))
    if not 0 <= count <= 4:
        raise ValueError("unsupported deliberate cell weight")
    assert len(set(cell)) == count
    return cell


def _dic_grid(rng, support, palette, majority):
    return [[_dic_cell(rng, count, palette, majority+i+j)
             for j, count in enumerate(row)]
            for i, row in enumerate(support)]


def _dic_component(G, side, grid):
    """Probe the verified lift; never implement a separate group action."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _dic_cycles(rows, width):
    """Exact four-cycles, a six-cycle proxy, and connected components.

    The triangle-minus-stars expression counts six-cycles exactly
    when four-cycles are absent; otherwise it is only a tie-breaker.
    """
    size = len(rows)
    adj = [0] * size
    degrees = [0] * width
    four = 0
    for i, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            degrees[bit.bit_length()-1] += 1
            bits ^= bit
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap-1) // 2
            if overlap:
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    triangles = 0
    for i in range(size):
        later = adj[i] >> (i+1)
        while later:
            bit = later & -later
            j = i + 1 + bit.bit_length() - 1
            triangles += ((adj[i] & adj[j]) >> (j+1)).bit_count()
            later ^= bit
    stars = sum(d*(d-1)*(d-2)//6 for d in degrees)

    unseen = (1 << size) - 1
    parts = 0
    while unseen:
        start = unseen & -unseen
        unseen ^= start
        todo = start
        parts += 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            neighbors = adj[bit.bit_length()-1] & unseen
            unseen ^= neighbors
            todo |= neighbors
    return four, max(0, triangles-stars), parts


def _dic_small_kernel(rows, width):
    """Exact existence test for component kernel weights one through four."""
    columns = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= tag
            row ^= bit
    if 0 in columns or len(set(columns)) != width:
        return True
    singles = set(columns)
    pairs = set()
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            if syndrome in singles or syndrome in pairs:
                return True
            pairs.add(syndrome)
    return False


def _dic_kernel_upper(rng, rows, width, trials):
    """Verified kernel words from randomized systematic bases.

    Returned minima are upper bounds, not distance certificates
    and not circuit-level failure estimates.
    """
    best = width
    for _ in range(trials):
        mat = list(rows)
        order = list(range(width))
        rng.shuffle(order)
        pivots = []
        rank = 0
        for c in order:
            bit = 1 << c
            hit = next((r for r in range(rank, len(mat))
                        if mat[r] & bit), None)
            if hit is None:
                continue
            mat[rank], mat[hit] = mat[hit], mat[rank]
            pivot_row = mat[rank]
            for r in range(len(mat)):
                if r != rank and mat[r] & bit:
                    mat[r] ^= pivot_row
            pivots.append(c)
            rank += 1
            if rank == len(mat):
                break
        occupied = set(pivots)
        basis = []
        for c in order:
            if c in occupied:
                continue
            bit = 1 << c
            word = bit
            for r, p in enumerate(pivots):
                if mat[r] & bit:
                    word ^= 1 << p
            basis.append(word)
            best = min(best, word.bit_count())
        for i, word in enumerate(basis):
            for j in range(i):
                best = min(best, (word ^ basis[j]).bit_count())
        if best <= 6:
            break
    return best


def _dic_candidates(rng, G, side, support, palette, majority):
    """Multistart optimization within one fixed architectural brief."""
    import math

    positions = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
    pool = []

    def analyze(grid):
        rows, width = _dic_component(G, side, grid)
        four, six, parts = _dic_cycles(rows, width)
        energy = (80*four + six/4.0)/G.q + 25*(parts-1)
        return energy, rows, width, four, six, parts

    def retain(grid, result):
        _, rows, width, four, six, parts = result
        bad = _dic_small_kernel(rows, width)
        deficit = len(rows) - gf2_rank(rows)
        key = (int(bad), parts-1, deficit, four, six)
        pool.append((key, _dic_copy(grid), rows, width))

    for restart in range(4):
        grid = _dic_grid(rng, support, palette, majority)
        current = analyze(grid)
        retain(grid, current)
        local_grid = _dic_copy(grid)
        local = current
        for step in range(140):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = _dic_cell(
                rng, support[i][j], palette, majority+i+j)
            proposed = analyze(grid)
            delta = proposed[0] - current[0]
            temperature = 15.0*(0.97**step) + 0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if current[0] < local[0]:
                    local_grid = _dic_copy(grid)
                    local = current
            else:
                grid[i][j] = old
        retain(local_grid, local)

    pool.sort(key=lambda entry: entry[0])
    # Exact repeated fillings are not additional finalists.
    finalists = []
    seen = set()
    for key, grid, rows, width in pool:
        signature = tuple(tuple(tuple(sorted(cell)) for cell in row)
                          for row in grid)
        if signature in seen:
            continue
        seen.add(signature)
        d = _dic_kernel_upper(
            rng, rows, width, 32 if side == "A" else 20)
        finalists.append((key, d, grid))
        if len(finalists) == 4:
            break
    return finalists


def NEW_CTX(rng, G, idx):
    """New C8-unit contracts with complete-code decoder-aware selection."""
    import math

    case, batch = idx % 20, idx // 20
    architecture, mode = divmod(case, 5)
    profiles = (
        ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
        ((1, 2), (1, 2), (2, 1), (2, 1), (2, 2)),
        ((2, 1), (2, 1), (2, 1), (2, 1), (0, 4)),
        ((3, 0), (0, 3), (1, 2), (1, 2), (2, 1)),
    )
    profile = profiles[architecture]
    sa = [[3, 3]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(11, W_MAX)

    q = G.q
    assert q == 48
    mask = (1 << q) - 1
    steps = {
        24: (1, 5, 7, 11, 13, 17, 19, 23),
        12: (2, 10, 14, 22),
        8: (3, 9, 15, 21),
        6: (4, 20),
        4: (6, 18),
    }
    portfolios = (
        ((24,), (24,), (12, 6), (24, 8), (24, 12)),
        ((12,), (24, 8), (24, 6), (12, 8), (24, 4)),
        ((24, 6), (12, 8), (24, 4), (24, 12), (12, 6)),
    )
    palette = portfolios[batch % 3][mode]
    a_order = (
        (24, 12, 24, 12, 24),
        (12, 24, 12, 24, 12),
        (24, 24, 12, 12, 24),
    )[batch % 3][mode]

    sites = [(i, j) for i in range(2) for j in range(5)
             if sb[i][j] == 2]
    start = (2 * mode + batch) % 5
    sites.sort(key=lambda p: ((p[1] - start) % 5,
                              p[0] ^ (batch % 2)))
    mixed_count = min((0, 1, 1, 2, 3)[mode], len(sites))
    mixed = set()
    used_columns = set()
    for site in sites:
        if len(mixed) == mixed_count:
            break
        if site[1] not in used_columns:
            mixed.add(site)
            used_columns.add(site[1])
    assert len(mixed) == mixed_count

    # Compose only singleton blocks obtained from the verified helper.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, value in enumerate(block):
                        out[t] ^= value << (j * q)
            rows.extend(out)
        return rows, len(grid[0]) * q

    def rotation_step(i, j):
        order = palette[(j + 2 * i + batch) % len(palette)]
        return rng.choice(steps[order])

    def draw_b_cell(i, j):
        weight = sb[i][j]
        if not weight:
            return []
        base = G.pick(rng)
        relative = [G.e]
        if weight == 2:
            g = ((rng.randrange(24), 1) if (i, j) in mixed
                 else (rotation_step(i, j), 0))
            relative.append(g)
        elif weight == 3:
            if (mode + batch) % 2 == 0:
                # Odd augmentation inside the normal C8 subgroup.
                u = rng.choice((1, 3, 5, 7))
                relative = [G.e, ((3*u) % 24, 0),
                            ((9*u) % 24, 0)]
            else:
                relative = [G.e, (rotation_step(i, j), 0),
                            (rng.randrange(24), 1)]
        elif weight == 4:
            if mode == 0:
                # A genuinely four-term, all-rotation cell. Exclude
                # involutory differences without presuming its rank.
                for attempt in range(128):
                    values = [0] + rng.sample(range(1, 24), 3)
                    if all((a-b) % 24 != 12
                           for t, a in enumerate(values)
                           for b in values[:t]):
                        break
                else:
                    raise ValueError("quartic rotation brief failed")
                relative = [(a, 0) for a in values]
            elif mode == 2:
                c = rng.randrange(24)
                relative = [
                    G.e, (rotation_step(i, j), 0),
                    (c, 1), ((c + rng.choice(steps[8])) % 24, 1),
                ]
            else:
                d = rotation_step(i, j)
                choices = [h for h in range(1, 24)
                           if h != d and h != 12
                           and (h-d) % 24 != 12]
                relative = [G.e, (d, 0), (rng.choice(choices), 0),
                            (rng.randrange(24), 1)]
        cell = [G.mul(base, g) for g in relative]
        assert len(cell) == weight
        assert len({G._key(g) for g in cell}) == weight
        return cell

    def draw_b():
        return [[draw_b_cell(i, j) for j in range(5)]
                for i in range(2)]

    # A0 is invertible: its three terms have odd augmentation in
    # F2[C8]. It is not the parent's mixed-coset trinomial model.
    unit = [G.e, (3, 0), (9, 0)]
    apool = []
    for d in steps[a_order]:
        if d > 12:
            continue
        for c in (0, 1):
            A = [[list(unit), [G.e, (d, 0), (c, 1)]]]
            rows, width = component(A, "A")
            assert gf2_rank([row & mask for row in rows]) == q
            four, six, parts = _dic_cycles(rows, width)
            bad = int(_dic_small_kernel(rows, width))
            # A common permutation stream compares actual kernel words.
            d_est = _dic_kernel_upper(
                random.Random(917039 + idx), rows, width, 40)
            key = (bad, parts-1, max(0, 10-d_est), four, six)
            apool.append((key, d_est, A, rows))
    apool.sort(key=lambda item: item[0])
    assert apool

    pool, seen = [], set()
    serial = 0

    def analyze_b(B):
        rows, width = component(B, "B")
        four, six, parts = _dic_cycles(rows, width)
        deficit = len(rows) - gf2_rank(rows)
        energy = ((100*four + six/8.0)/q
                  + 30*deficit + 40*(parts-1))
        return energy, rows, width, four, six, parts, deficit

    def offer_b(B, state):
        nonlocal serial
        _, rows, width, four, six, parts, deficit = state
        if deficit:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        serial += 1
        key = (parts-1, four, six)
        pool.append((key, serial, _dic_copy(B), list(rows), width))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    mutable = [(i, j) for i in range(2) for j in range(5)
               if sb[i][j]]
    for restart in range(4):
        B = draw_b()
        current = analyze_b(B)
        offer_b(B, current)
        for step in range(180 + 20*(batch % 3)):
            i, j = rng.choice(mutable)
            old = B[i][j]
            if rng.random() < 0.30:
                shift = G.pick(rng)
                B[i][j] = [G.mul(shift, g) for g in old]
            else:
                B[i][j] = draw_b_cell(i, j)
            proposed = analyze_b(B)
            delta = proposed[0] - current[0]
            temperature = 10.0*(0.982**step) + 0.10
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if step % 12 == 0:
                    offer_b(B, current)
            else:
                B[i][j] = old
        offer_b(B, current)

    if not pool:
        for attempt in range(128):
            B = draw_b()
            offer_b(B, analyze_b(B))
            if len(pool) >= 4:
                break
    if not pool:
        raise ValueError("B contract missed full row rank")

    bpool = []
    for coarse, serial, B, rows, width in pool:
        actual, actual_width = _dic_component(G, "B", B)
        assert actual == rows and actual_width == width
        bad = int(_dic_small_kernel(rows, width))
        d_est = _dic_kernel_upper(
            random.Random(918043 + idx), rows, width, 28)
        key = (bad, coarse[0], max(0, 10-d_est),
               coarse[1], coarse[2])
        bpool.append((key, d_est, B, rows))
    bpool.sort(key=lambda item: item[0])

    # Coarse protection is capped. Larger sampled upper bounds are
    # not rewarded indefinitely and are never called certificates.
    options = []
    for ak, ad, A, ar in apool:
        for bk, bd, B, br in bpool:
            key = (
                ak[0] + bk[0], ak[1] + bk[1],
                max(0, 10-ad) + max(0, 10-bd),
                -min(ad, bd, 12), ak[3] + bk[3],
                ak[4] + bk[4],
            )
            options.append((key, A, B, ar, br))
    options.sort(key=lambda item: item[0])

    verified = []
    fallback = None
    for geometry, A, B, ar, br in options:
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 576 or n-gf2_rank(hx)-gf2_rank(hz) != 144:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(11, W_MAX)
        actual_a = [
            (row & mask) | (((row >> (5*q)) & mask) << q)
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
        assert actual_a == ar and actual_b == br
        cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
        geometry = geometry + (
            max(cx/len(hx), cz/len(hz)), cx+cz)
        entry = (geometry, A, B, hx, hz)
        if fallback is None:
            fallback = entry
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        verified.append(entry)
        if len(verified) == 8:
            break
    if not verified:
        if fallback is None:
            raise ValueError("no C8-unit pair has verified k144")
        verified = [fallback]
    verified.sort(key=lambda item: item[0])

    try:
        import numpy as np
    except ImportError:
        np = None

    def probe(checks, stabilizers, seed, shots):
        """Stabilizer-aware data-noise surrogate, NOT ASC circuit pL.

        Shared random streams compare candidates. Opposite-check pairs
        mark possible ancilla-spread opportunities, not scheduled hooks.
        Unsettled shots count as probe failures.
        """
        if np is None:
            return (0, 0, 0)
        n = 576

        def neighborhoods(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length()-1)
                    row ^= bit
                result.append(ns)
            return result

        ns = neighborhoods(checks)
        mate = neighborhoods(stabilizers)
        degree = np.zeros(n, dtype=float)
        mate_degree = np.zeros(n, dtype=float)
        for row in ns:
            degree[row] += 1
        for row in mate:
            mate_degree[row] += 1
        exposure = degree + mate_degree
        channel = np.random.default_rng(seed)
        levels = np.asarray((0.014, 0.024, 0.034))
        rates = (exposure[:, None]/exposure.mean()
                 * levels[np.arange(shots) % 3][None, :])
        errors = (channel.random((n, shots)) < rates).astype(np.uint8)

        # The random selectors are shared; their selected neighborhoods
        # necessarily depend on each candidate's actual stabilizers.
        hook_rows = channel.integers(len(mate), size=shots)
        first = channel.random(shots)
        second = channel.random(shots)
        hooks = channel.random(shots) < 0.22
        for shot in np.flatnonzero(hooks):
            row = mate[int(hook_rows[shot])]
            a = min(int(first[shot]*len(row)), len(row)-1)
            b = min(int(second[shot]*(len(row)-1)), len(row)-2)
            b += int(b >= a)
            errors[row[a], shot] ^= 1
            errors[row[b], shot] ^= 1
        marginal = 0.44*mate_degree/mate_degree.sum()
        prior_rates = np.minimum(rates + marginal[:, None], 0.25)
        prior = np.log((1-prior_rates)/prior_rates)

        variables, row_edges = [], []
        var_edges = [[] for _ in range(n)]
        for row in ns:
            edges = []
            for c in row:
                edge = len(variables)
                variables.append(c)
                edges.append(edge)
                var_edges[c].append(edge)
            row_edges.append(edges)
        ne = len(variables)
        variables = np.asarray(variables, dtype=np.intp)
        rd = max(map(len, row_edges))
        vd = max(map(len, var_edges))
        rmap = np.full((len(ns), rd), ne, dtype=np.intp)
        incidence = np.full((len(ns), rd), n, dtype=np.intp)
        vmap = np.full((n, vd), ne, dtype=np.intp)
        for r, edges in enumerate(row_edges):
            rmap[r, :len(edges)] = edges
            incidence[r, :len(edges)] = ns[r]
        for c, edges in enumerate(var_edges):
            vmap[c, :len(edges)] = edges
        valid = rmap != ne
        padded = np.zeros((n+1, shots), dtype=np.uint8)
        padded[:n] = errors
        target = (padded[incidence].sum(axis=1) & 1).astype(np.uint8)
        target_sign = np.where(target != 0, -1.0, 1.0)

        incoming = np.empty((ne+1, shots), dtype=float)
        incoming[:ne] = prior[variables]
        incoming[ne] = 1e6
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        slots = np.arange(rd)[None, :, None]
        iteration_sum = 0

        for iteration in range(1, 33):
            gathered = incoming[rmap]
            signs = np.where(gathered >= 0, 1.0, -1.0)
            absolute = np.abs(gathered)
            smallest_at = np.argmin(absolute, axis=1)
            smallest = np.min(absolute, axis=1)
            second_min = np.partition(absolute, 1, axis=1)[:, 1, :]
            excluded = np.where(
                slots == smallest_at[:, None, :],
                second_min[:, None, :], smallest[:, None, :])
            parity = np.prod(signs, axis=1)*target_sign
            outgoing = 0.80*excluded*signs*parity[:, None, :]
            messages[rmap[valid]] = outgoing[valid]
            messages[ne] = 0.0
            posterior = prior + messages[vmap].sum(axis=1)
            hard = (posterior < 0).astype(np.uint8)

            if iteration % 4 == 0:
                padded[:n] = hard
                syndrome = (padded[incidence].sum(axis=1) & 1)
                settled = np.all(syndrome == target, axis=0) & active
                final[:, settled] = hard[:, settled]
                iteration_sum += iteration*int(settled.sum())
                active[settled] = False
                if not active.any():
                    break
            extrinsic = posterior[variables] - messages[:ne]
            incoming[:ne] = (
                0.25*incoming[:ne]
                + 0.75*np.clip(extrinsic, -24.0, 24.0))
            incoming[ne] = 1e6

        final[:, active] = hard[:, active]
        unresolved = int(active.sum())
        iteration_sum += 33*unresolved

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        failed = unresolved
        residual = errors ^ final
        for shot in np.flatnonzero(~active):
            word = 0
            for c in np.flatnonzero(residual[:, shot]):
                word |= 1 << int(c)
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    failed += 1
                    break
                word ^= old
        return int(failed), unresolved, int(iteration_sum)

    def probe_key(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    trained = []
    seed = 120480025 + 1009*idx
    for geometry, A, B, hx, hz in verified[:5]:
        # X residuals are checked by HZ and quotiented by HX;
        # Z residuals are checked by HX and quotiented by HZ.
        x = probe(hz, hx, seed, 96)
        z = probe(hx, hz, seed+1, 96)
        key = probe_key(x, z) + geometry
        trained.append((key, geometry, A, B, hx, hz, x, z))
    trained.sort(key=lambda item: item[0])

    finalists = []
    for train_key, geometry, A, B, hx, hz, tx, tz in trained[:2]:
        # Independent validation avoids selecting solely for one small
        # shared stress sample. It remains an uncalibrated surrogate.
        vx = probe(hz, hx, seed+104729, 192)
        vz = probe(hx, hz, seed+104730, 192)
        quality = probe_key(vx, vz) + train_key
        finalists.append((quality, A, B, hx, hz, tx, tz, vx, vz))
    finalists.sort(key=lambda item: item[0])
    quality, A, B, hx, hz, tx, tz, vx, vz = finalists[0]
    cnots = 6*sum(row.bit_count() for row in hx+hz)
    assert cnots == (42 + 3*sum(map(sum, sb)))*q*6
    assert cnots in (25056, 25920)
    return {
        "idx": idx, "contract": (architecture, mode, batch),
        "k": 144, "screen": quality,
        "probe": (tx, tz, vx, vz), "cnots": cnots,
        "cells": {"A": A, "B": B},
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the jointly selected complete cell without late resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Higher k at equal d2s necessarily improves circuit score: current-parent slot10 had k=150,d2s=10,s=5.748, versus slot25 k=144,d2s=10,s=6.280; slot10 also required 29952 rather than 27936 CNOTs.
# REFUTED: Long-ratio component screening and equal d2s/depth ensure comparable basis balance: Dic12 slots3/4 both had k144,d2s12,depth14, but failed 73/nine X shots and 15/four Z shots per200000, scoring 6.275/7.005.
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
