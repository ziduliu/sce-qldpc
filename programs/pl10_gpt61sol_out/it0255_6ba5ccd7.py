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
CONCEPT = ("For D14 x C3, order 42, A=(1,2), B=(2,6), rank-verified unit trinomials with order-six/seven internal ratios and joint stabilizer-aware decoder screening should improve the 0.300 frontier beyond fitness 6.1594; measured ASC pL can refute this.")
GROUP = ("Jump from the order-39 Frobenius group to D14 x C3. This direct product has centre order three and 15 conjugacy classes; its order-42 bed is absent from BEDS. The normal cyclic subgroup C7 x C3 has order 21 and supplies long binomial ratios. Related order-42 beds motivate a substantial nonabelian lift, but establish no advantage for this group. Trinomial invertibility is checked on the verified lift, not inferred from odd augmentation.")
SHAPES = ("Retain (1,2)x(2,6) for cell 0.300. The size factor fourteen permits q<=42; q=42 gives n=588, kb=168 and guaranteed rate 2/7. Components have 84 variables/42 checks and 252 variables/84 checks. Require complete verified k=168. Parent weight-ten witnesses and collective failures covered both A columns and many B columns; weak, shifting Acol/Brow check enrichments do not establish a localized repair target.")
FILLING = ("Twenty contracts cross four canonical unit-trinomial pair architectures with five B ratio schedules using orders 21,7,6. Additional harness batches change B zero/three-term topology, not saved-code labels. B rows total nine and columns three, giving HX/HZ weights 9/12 and 25704 six-round CNOTs, versus the parent's 23868. Helper-derived component actions screen ranks, short kernels, capped sampled witnesses and triple boundaries. Complete finalists receive common-random-number BP probes whose settled residuals are classified modulo actual stabilizers. These probes omit ASC ordering, measurement histories and the relay cascade, so are selection surrogates rather than circuit measurements. Target the parent's Z failures reaching 100 shots at 77250..159250 samples, compared with best-slot 33/200000, despite identical k156,d2s10,w12,depth15.")
FAMILY_NAME = "r030_D14xC3_q42_12x26_unit_catalogue_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("D14 x C3 requires order 42")
    # (a,b,c) = x^a y^b z^c.
    # x^7=y^2=z^3=1, y*x*y^-1=x^-1, z central.
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 2, int(g[2]) % 3)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d)) % 7,
                (b + e) % 2, (c + f) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((-a if b == 0 else a) % 7, b, (-c) % 3)

    def idx(g):
        a, b, c = red(g)
        return a + 7*b + 14*c

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


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


def _jump_columns(rows, width):
    """Transpose only verified or verified-singleton-composed rows."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _jump_bp(checks, stabilizers, n, seed, shots=64):
    """Data-noise BP surrogate, NOT an ASC memory experiment.

    Common random streams include degree-weighted noise and pairs
    sharing an opposite-basis check. These pairs are not ASC-certified
    hooks. Settled residuals are classified against actual stabilizers.
    """
    import numpy as np

    def neighborhoods(rows):
        out = []
        for row in rows:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            out.append(ns)
        return out

    ns = neighborhoods(checks)
    weight = len(ns[0])
    assert all(len(row) == weight for row in ns)
    incidence = np.asarray(ns, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(checks), len(flat)
    degrees = np.bincount(flat, minlength=n)
    assert np.all(degrees > 0)
    vedges = np.full((n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        vedges[variable, used[variable]] = edge
        used[variable] += 1

    opposite = neighborhoods(stabilizers)
    opposite_degree = np.zeros(n, dtype=np.int32)
    for row in opposite:
        opposite_degree[row] += 1
    exposure = degrees + opposite_degree
    channel = np.random.default_rng(seed)
    levels = np.asarray((0.010, 0.020, 0.032))
    rates = levels[np.arange(shots) % 3, None] * exposure[None, :] / 6
    rates = np.clip(rates, 0.001, 0.20)
    error = channel.random((shots, n)) < rates
    for shot in range(shots):
        if channel.random() < 0.25:
            row = opposite[int(channel.integers(len(opposite)))]
            a, b = channel.choice(len(row), 2, replace=False)
            error[shot, row[int(a)]] ^= True
            error[shot, row[int(b)]] ^= True

    prior = np.log((1-rates)/rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    failures = work = 0
    for iteration in range(1, 25):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logs = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
        products = np.minimum(
            np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-7)
        magnitude = np.log1p(products)-np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)
        padded = np.zeros((count, edges+1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, vedges].sum(axis=2)
        hard = posterior < 0
        syndrome = (
            hard[:, flat].reshape(count, nr, weight).sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failures += int(nontrivial(word))
            work += iteration
        keep = ~settled
        if not np.any(keep):
            return failures, 0, work
        messages = np.clip(
            0.35*messages + 0.65*(posterior[:, flat]-outgoing),
            -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]
    unresolved = len(error)
    return failures+unresolved, unresolved, work+25*unresolved


def NEW_CTX(rng, G, idx):
    """Canonical A architectures, typed B ratios, then joint decoder selection."""
    import math
    from itertools import combinations, combinations_with_replacement

    assert G.q == 42
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    q, mask = G.q, (1 << G.q)-1
    case, batch = idx % 20, idx // 20
    architecture, policy = divmod(case, 5)

    # Extra batches introduce new support topologies at the same budget.
    first = (
        (1, 1, 1, 2, 2, 2),
        (0, 3, 1, 2, 2, 1),
        (0, 0, 3, 3, 1, 2),
    )[batch % 3]
    sa = [[3, 3]]
    sb = [list(first), [3-t for t in first]]
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 9 for row in sb)
    assert all(sum(col) == 3 for col in zip(*sb))
    assert sum(sa[0])+3 == 9
    assert 9+max(sa[0]) == 12
    assert _grid_weights(sa, sb) == 12 <= W_MAX

    # Every regular-action block comes from the verified helper.
    actions = {"A": {}, "B": {}}
    zero = [[[]]]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], zero)
        _, hz, _ = lifted_matrices(G, zero, [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for row in grid:
            block_rows = [0]*q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        block_rows[r] ^= word << (j*q)
            rows.extend(block_rows)
        return rows, len(grid[0])*q

    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = order
                break
        assert G.idx(g) in orders
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (6, 7, 21)
    }
    assert all(palettes.values())

    def normalized(cell):
        # Remove cell origins, but never independently transform
        # complete submitted codes into purported new candidates.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(origin))) for g in cell))
            for origin in cell)

    # Exhaust the permitted trinomial geometries before using them.
    # An empty unit subclass cannot cause a random palette-exhaustion slot.
    unit_map = {}
    usable = [g for g in G.elements if orders[G.idx(g)] in (6, 7, 21)]
    tested = set()
    for u, v in combinations(usable, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        difference = G.mul(G.inv(u), v)
        if orders[G.idx(difference)] not in (6, 7, 21):
            continue
        signature = normalized([G.e, u, v])
        if signature in tested:
            continue
        tested.add(signature)
        cell = [G.elements[t] for t in signature]
        rows, _ = component([[cell]], "A")
        if gf2_rank(rows) == q:
            unit_map[signature] = cell
    units = [unit_map[key] for key in sorted(unit_map)]
    if not units:
        raise ValueError("verified unit-trinomial catalogue is empty")

    # Genuine automorphisms of D14 x C3:
    # x->x^u, y->x^v*y, z->z^w.
    # Quotient cell origins and A-column exchange as well.
    # This is an internal-geometry filter, not a general Tanner classifier.
    automorphisms = [(u, v, w) for u in range(1, 7)
                     for v in range(7) for w in (1, 2)]
    images = []
    for cell in units:
        table = []
        for u, v, w in automorphisms:
            changed = [((u*g[0]+v*g[1]) % 7,
                        g[1], (w*g[2]) % 3) for g in cell]
            table.append(normalized(changed))
        images.append(table)

    def pair_identity(a, b):
        return min(tuple(sorted((images[a][t], images[b][t])))
                   for t in range(len(automorphisms)))

    pairs = list(combinations_with_replacement(range(len(units)), 2))
    catalogue_rng = random.Random(4200713)
    catalogue_rng.shuffle(pairs)
    coarse_bank, seen = [], set()
    for a, b in pairs:
        identity = pair_identity(a, b)
        if identity in seen:
            continue
        seen.add(identity)
        grid = [[list(units[a]), list(units[b])]]
        rows, width = component(grid, "A")
        four, six, parts = _dic_cycles(rows, width)
        bad = int(_dic_small_kernel(rows, width))
        coarse_bank.append(((bad, parts-1, four, six, identity),
                            grid, rows, width))
        if len(coarse_bank) >= 96:
            break
    coarse_bank.sort(key=lambda item: item[0])

    def triple_statistics(cols, exact_six=False):
        """Boundary exposure; exact six-word census only after <=4 exclusion."""
        histogram = {} if exact_six else None
        collisions = burden = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    weight = syndrome.bit_count()
                    if weight <= 3:
                        burden += (5-weight)**2
                    if histogram is not None:
                        old = histogram.get(syndrome, 0)
                        collisions += old
                        histogram[syndrome] = old+1
        if exact_six:
            # All columns have odd degree three. Shorter kernels have
            # been excluded, so matching distinct triples are disjoint.
            assert collisions % 10 == 0
        return collisions//10 if exact_six else 0, burden

    bank = []
    for coarse, grid, rows, width in coarse_bank[:24]:
        cols = _jump_columns(rows, width)
        assert all(col.bit_count() == 3 for col in cols)
        words6, burden = triple_statistics(cols, not coarse[0])
        if coarse[0]:
            words6 = 10**9
        sample_rng = random.Random(4200719)
        distance = _dic_kernel_upper(sample_rng, rows, width, 20)
        key = (coarse[0], words6, max(0, 10-distance),
               coarse[2], coarse[3], burden, coarse[4])
        bank.append((key, grid))
    bank.sort(key=lambda item: item[0])
    if len(bank) < 4:
        raise ValueError("fewer than four canonical unit-pair architectures")
    selected = (architecture+4*(batch//3)) % len(bank)
    akey, A = bank[selected]
    ar, awidth = component(A, "A")
    verified_a, verified_width = _dic_component(G, "A", A)
    assert ar == verified_a and awidth == verified_width

    sites = [(i, j) for i in range(2) for j in range(6)
             if sb[i][j] == 2]
    sites.sort(key=lambda p: ((p[1]-architecture-2*batch) % 6, p[0]))
    schedule = {site: 21 for site in sites}
    if policy == 1:
        schedule[sites[0]] = 6
    elif policy == 2:
        for site in sites[:2]:
            schedule[site] = 6
    elif policy == 3:
        for site in sites[:2]:
            schedule[site] = 7
    elif policy == 4:
        schedule[sites[0]] = 7
        if len(sites) >= 3:
            schedule[sites[2]] = 6

    def draw_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        anchor = G.pick(rng)
        if count == 1:
            relative = [G.e]
        elif count == 2:
            relative = [G.e, rng.choice(palettes[schedule[i, j]])]
        else:
            relative = rng.choice(units)
        cell = [G.mul(anchor, g) for g in relative]
        assert len(cell) == count and len(set(cell)) == count
        return cell

    def draw_B():
        return [[draw_cell(i, j) for j in range(6)] for i in range(2)]

    def analyze(B):
        rows, width = component(B, "B")
        four, six, parts = _dic_cycles(rows, width)
        deficit = len(rows)-gf2_rank(rows)
        energy = (100*deficit+40*(parts-1)
                  +(100*four+six/8.0)/q)
        return energy, rows, width, deficit, parts-1, four, six

    pool, identities = [], set()

    def retain(B, result):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
        if signature in identities:
            return
        identities.add(signature)
        _, rows, width, deficit, parts, four, six = result
        key = (deficit, parts, four, six, signature)
        pool.append((key, _dic_copy(B), list(rows), width))
        pool.sort(key=lambda item: item[0])
        del pool[10:]

    positions = [(i, j) for i in range(2) for j in range(6) if sb[i][j]]
    for restart in range(4):
        B = draw_B()
        current = analyze(B)
        retain(B, current)
        for step in range(180):
            i, j = rng.choice(positions)
            old = B[i][j]
            B[i][j] = draw_cell(i, j)
            proposed = analyze(B)
            delta = proposed[0]-current[0]
            temperature = 14.0*(0.982**step)+0.12
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if step % 12 == 0:
                    retain(B, current)
            else:
                B[i][j] = old
        retain(B, current)

    def sample_words(cols):
        """Actual kernel witnesses; the minimum remains only an upper bound."""
        srng = random.Random(4200791+65537*idx)
        order = list(range(len(cols)))
        minimum = len(cols)+1
        short = set()
        for trial in range(32):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 10 and len(short) < 8192:
                        short.add(word)
                    for other in basis[:i]:
                        value = word ^ other
                        if not value:
                            continue
                        w = value.bit_count()
                        minimum = min(minimum, w)
                        if w <= 10 and len(short) < 8192:
                            short.add(value)
                        if w < weight:
                            word, weight = value, w
                    basis[i] = word
            if minimum <= 4:
                break
        return minimum, sum(w.bit_count() <= 8 for w in short), len(short)

    def anchored_boundary(cols):
        loads = []
        for anchor in range(0, len(cols), q):
            burden = 0
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 3:
                        burden += (5-weight)**2
            loads.append(burden)
        return max(loads), sum(loads)/len(loads)

    refined = []
    for coarse, B, rows, width in pool[:8]:
        verified_b, verified_width = _dic_component(G, "B", B)
        assert rows == verified_b and width == verified_width
        cols = _jump_columns(rows, width)
        assert all(col.bit_count() == 3 for col in cols)
        bad = int(_dic_small_kernel(rows, width))
        distance, low8, low10 = sample_words(cols)
        peak, mean = anchored_boundary(cols)
        key = (bad, coarse[1], coarse[0], max(0, 10-distance),
               low8, low10, peak, mean, coarse[2], coarse[3])
        refined.append((key, B))
    refined.sort(key=lambda item: item[0])

    options = []
    for bkey, B in refined:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 588 or k != 168:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12
        assert 6*sum(row.bit_count() for row in hx+hz) == 25704
        unsafe = int(_design_small_logical(hx, hz, n)
                     or _design_small_logical(hz, hx, n))
        geometry = (unsafe,)+bkey
        options.append((geometry, B, hx, hz))
    options.sort(key=lambda item: item[0])

    # Legal rescue retains the prescribed support and ratio schedule.
    if not options:
        for attempt in range(256):
            B = draw_B()
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n == 588 and k == 168:
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                options.append(((1, 10**9, attempt), B, hx, hz))
                break
    if not options:
        raise ValueError("no exact-bound direct-product filling")

    finalists = []
    seed = 4200817+1009*idx
    for geometry, B, hx, hz in options[:4]:
        x = _jump_bp(hz, hx, 588, seed)
        z = _jump_bp(hx, hz, 588, seed+1)
        quality = (
            geometry[0], max(x[0], z[0]), x[0]+z[0],
            max(x[1], z[1]), x[1]+z[1], x[2]+z[2],
        )+geometry[1:]
        finalists.append((quality, B, x, z))
    finalists.sort(key=lambda item: item[0])
    quality, B, x, z = finalists[0]
    assert lifted_k(G, A, B) == 168
    assert all(len(cell) <= 5 and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {
        "idx": idx, "k": 168,
        "contract": (selected, policy, batch),
        "a_architecture": akey,
        "bp_surrogate": (x, z), "screen": quality,
        "cells": {"A": A, "B": B},
    }


def _legacy_frob39_ctx(rng, G, idx):
    """Inactive historical builder; the direct-product NEW_CTX is authoritative."""
    import math

    case, batch = idx % 20, idx // 20
    geometry, profile_id = divmod(case, 5)
    # Undirected rotation ratios modulo column exchange/inversion:
    # 1; {2,6}; {3,4}; {5}. These are structural search briefs,
    # not a complete classification of Tanner-graph isomorphisms.
    displacement = (1, 2, 3, 5)[geometry]
    profiles = (
        ((1,2), (1,2), (1,2), (2,1), (2,1), (2,1)),
        ((0,3), (3,0), (1,2), (1,2), (2,1), (2,1)),
        ((0,3), (0,3), (3,0), (3,0), (1,2), (2,1)),
        ((0,3), (1,2), (1,2), (1,2), (3,0), (3,0)),
        ((1,3), (3,1), (1,1), (1,1), (1,2), (2,1)),
    )
    profile = profiles[profile_id]
    sa = [[3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    wx = 6 + max(map(sum, profile))
    wz = max(map(sum, sb)) + 3
    assert G.q == 39 and sum(map(sum, sb)) == 18
    assert max(wx, wz) == _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        result = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j*q)
            result.extend(out)
        return result, len(grid[0])*q

    def columns(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return cols

    # B's internal order contract changes across geometry and extra
    # harness batches. Whole-cell voltages are then optimized freely.
    sites = [(i, j) for i in range(2) for j in range(6)
             if sb[i][j] == 2]
    sites.sort(key=lambda p: (
        (p[1]-geometry-2*batch) % 6, p[0] ^ (batch % 2)))
    mixed = set(sites[:min(len(sites), (geometry+batch) % 4)])

    def b_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        base = G.pick(rng)
        relative = [G.e]
        if count == 2:
            if (i, j) in mixed:
                relative.append((rng.randrange(13), 1+rng.randrange(2)))
            else:
                relative.append((rng.randrange(1, 13), 0))
        elif count == 3:
            relative.extend((
                (rng.randrange(1, 13), 0),
                (rng.randrange(13), 1+(i+j+geometry+batch) % 2),
            ))
        cell = [G.mul(base, g) for g in relative]
        assert len({G._key(g) for g in cell}) == count
        return cell

    def draw_B():
        return [[b_cell(i, j) for j in range(6)] for i in range(2)]

    def coarse(grid, side):
        rows, width = component(grid, side)
        four, six, parts = _dic_cycles(rows, width)
        deficit = len(rows)-gf2_rank(rows)
        energy = ((120*four + six/8.0)/q
                  + 40*(parts-1) + 60*deficit)
        return energy, rows, width, four, six, parts, deficit

    def shortlist(side):
        pool, seen = [], set()
        serial = 0

        def retain(grid, result):
            nonlocal serial
            _, rows, width, four, six, parts, deficit = result
            if deficit:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append(((parts-1, four, six, serial),
                         _dic_copy(grid), list(rows), width))
            pool.sort(key=lambda entry: entry[0])
            del pool[6:]

        if side == "A":
            # Normalize the first complementary offset by conjugation.
            # The second displacement remains the contract's geometry.
            complement = 1+(geometry+batch) % 2
            parameters = list(range(13))
            rng.shuffle(parameters)
            for t in parameters:
                grid = [[
                    [G.e, (1, 0), (0, 1)],
                    [G.e, (displacement, 0), (t, complement)],
                ]]
                retain(grid, coarse(grid, side))
        else:
            positions = [(i, j) for i in range(2) for j in range(6)
                         if sb[i][j]]
            for restart in range(4):
                grid = draw_B()
                current = coarse(grid, side)
                retain(grid, current)
                for step in range(200):
                    i, j = rng.choice(positions)
                    old = grid[i][j]
                    grid[i][j] = b_cell(i, j)
                    proposed = coarse(grid, side)
                    delta = proposed[0]-current[0]
                    temperature = 14.0*(0.982**step)+0.10
                    if delta <= 0 or rng.random() < math.exp(
                            -min(700.0, delta/temperature)):
                        current = proposed
                        if step % 10 == 0:
                            retain(grid, current)
                    else:
                        grid[i][j] = old
                retain(grid, current)
            if not pool:
                for _ in range(96):
                    grid = draw_B()
                    retain(grid, coarse(grid, side))
                    if pool:
                        break
        if not pool:
            raise ValueError("odd-order contract missed full-rank component")
        return pool

    def boundary(cols):
        """Anchored small triple boundaries, not a circuit estimate."""
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, value in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ value
                for j in range(i):
                    if j != anchor:
                        weight = (pair ^ cols[j]).bit_count()
                        if weight <= 4:
                            cost += (5-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)/len(loads)

    def sampled(cols, side):
        """Verified kernel witnesses from common randomized bases.

        Minima are upper bounds; multiplicity is sampled. Neither
        quantity is treated as the scored circuit failure probability.
        """
        srng = random.Random(398117+65537*idx+(side == "B"))
        order = list(range(len(cols)))
        best = len(cols)+1
        short = set()
        for trial in range(48 if side == "A" else 32):
            srng.shuffle(order)
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
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    best = min(best, weight)
                    if weight <= 12 and len(short) < 8192:
                        short.add(word)
                    for other in basis[:i]:
                        value = word ^ other
                        if not value:
                            continue
                        w = value.bit_count()
                        best = min(best, w)
                        if w <= 12 and len(short) < 8192:
                            short.add(value)
                        if w < weight:
                            word, weight = value, w
                    basis[i] = word
            if best <= 6:
                break
        return best, sum(v.bit_count() <= 10 for v in short), len(short)

    def refine(side, pool):
        entries = []
        for coarse_key, grid, rows, width in pool:
            verified, verified_width = _dic_component(G, side, grid)
            assert rows == verified and width == verified_width
            bad = int(_dic_small_kernel(rows, width))
            cols = columns(rows, width)
            d, low10, low12 = sampled(cols, side)
            peak, mean = boundary(cols)
            key = (bad, coarse_key[0], max(0, 10-d),
                   low10, peak, mean, low12,
                   -min(d, 12), coarse_key[1], coarse_key[2])
            entries.append((key, grid))
        entries.sort(key=lambda entry: entry[0])
        return entries

    apool = refine("A", shortlist("A"))
    bpool = refine("B", shortlist("B"))
    options = []
    for ak, A in apool:
        for bk, B in bpool:
            # Both components must meet the same capped floor; thereafter
            # balance ambiguity and boundary exposure before cycle ties.
            key = (ak[0]+bk[0], ak[1]+bk[1], ak[2]+bk[2],
                   max(6*ak[3], 2*bk[3]), 6*ak[3]+2*bk[3],
                   max(ak[4], bk[4]), ak[5]+bk[5],
                   6*ak[6]+2*bk[6], max(ak[7], bk[7]),
                   ak[8]+bk[8], ak[9]+bk[9])
            options.append((key, A, B))
    options.sort(key=lambda entry: entry[0])

    verified_options = []
    for key, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 546 or k != 156:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
        assert 6*sum(row.bit_count() for row in hx+hz) == 23868
        cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
        verified_options.append((
            key+(max(cx/len(hx), cz/len(hz)), cx+cz),
            A, B, hx, hz))
        if len(verified_options) == 8:
            break
    verified_options.sort(key=lambda entry: entry[0])

    fallback = None
    for key, A, B, hx, hz in verified_options:
        candidate = {
            "idx": idx, "case": case, "k": 156, "screen": key,
            "contract": (displacement, profile_id, batch),
            "cells": {"A": A, "B": B},
        }
        if fallback is None:
            fallback = candidate
        if (_design_small_logical(hx, hz, 546)
                or _design_small_logical(hz, hx, 546)):
            continue
        return candidate
    if fallback is not None:
        return fallback
    raise ValueError("no verified k=156 pair in odd-order contract")


def _legacy_dic12_ctx(rng, G, idx):
    """Unused previous builder; NEW_CTX above is the sole slot entry point."""
    import math

    case, batch = idx % 20, idx // 20
    architecture, mode = divmod(case, 5)
    contracts = (
        ((3, 3), ((1, 1, 1, 2, 2), (2, 2, 2, 1, 1))),
        ((3, 3), ((1, 1, 2, 2, 2), (2, 2, 1, 1, 2))),
        ((3, 3), ((1, 1, 1, 2, 3), (2, 2, 2, 1, 1))),
        ((3, 4), ((1, 1, 1, 2, 2), (2, 2, 2, 1, 1))),
    )
    aw, bw = contracts[architecture]
    sa, sb = [list(aw)], [list(row) for row in bw]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    mask = (1 << q) - 1
    steps = {
        24: (1, 5, 7, 11, 13, 17, 19, 23),
        12: (2, 10, 14, 22),
        8: (3, 9, 15, 21),
    }
    a_orders = (
        (8, 8), (8, 12), (8, 24), (8, 8), (8, 24),
    )[mode]
    if batch:
        a_orders = (8, (8, 12, 24)[(mode + batch) % 3])
    b_orders = (
        (24,), (12,), (24, 12), (8, 24), (8, 12, 24),
    )[mode]

    # Indexed mixed-ratio positions are part of each contract.
    # Additional harness batches change positions and order portfolios,
    # rather than replay an optimized filling's labeling orbit.
    sites = [(i, j) for i in range(2) for j in range(5)
             if sb[i][j] == 2]
    start = (mode + 2 * batch) % 5
    sites.sort(key=lambda p: ((p[1] - start) % 5,
                              p[0] ^ ((mode + batch) % 2)))
    number_mixed = (1, 1, 2, 2, 3)[mode]
    mixed = set()
    used_columns = set()
    for site in sites:
        if site[1] not in used_columns:
            mixed.add(site)
            used_columns.add(site[1])
        if len(mixed) == number_mixed:
            break
    assert len(mixed) == number_mixed

    # Cache only actions returned by the verified construction helper.
    # XOR composition below introduces no separate lift convention.
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
                    term = actions[side][G.idx(g)]
                    shift = j * q
                    for t, bits in enumerate(term):
                        out[t] ^= bits << shift
            rows.extend(out)
        return rows, len(grid[0]) * q

    def draw_cell(side, i, j):
        if side == "A":
            d = rng.choice(steps[a_orders[j]])
            # A0 lies in <x^3,y>, a two-group, with odd augmentation.
            # A1 uses a different reflection residue to couple the
            # Sylow-subgroup geometries rather than repeat one component.
            residue = 0 if j == 0 else 1 + (mode + batch) % 2
            c = rng.choice([t for t in range(24) if t % 3 == residue])
            cell = [G.e, (d, 0), (c, 1)]
            if aw[j] == 4:
                choices = [t for t in range(1, 24)
                           if t != d and t != 12
                           and (t - d) % 24 != 12]
                cell.append((rng.choice(choices), 0))
        else:
            weight = sb[i][j]
            base = G.pick(rng)
            if weight == 1:
                cell = [base]
            elif weight == 2 and (i, j) in mixed:
                # Every element outside <x> in Dic12 has order four.
                relative = (rng.randrange(24), 1)
                cell = [base, G.mul(base, relative)]
            else:
                order = b_orders[(j + 2 * i + batch) % len(b_orders)]
                if weight == 3:
                    order = 8 if (mode + batch) % 2 == 0 else 24
                relative = [(0, 0), (rng.choice(steps[order]), 0)]
                if weight == 3:
                    relative.append((rng.randrange(24), 1))
                cell = [G.mul(base, g) for g in relative]
        assert len({G._key(g) for g in cell}) == len(cell)
        return cell

    def draw_grid(side):
        support = sa if side == "A" else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(support)]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def boundary(cols):
        """Anchored triple exposure, not an exact trapping-set census."""
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, value in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ value
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        cost += (5 - weight) ** 2
            loads.append(cost)
        return max(loads), sum(loads) / len(loads)

    def a_spectrum(cols):
        """Exact weight-five/six counts, conditional on no words <=4."""
        pairs = {cols[i] ^ cols[j]
                 for i in range(len(cols)) for j in range(i)}
        triples = {}
        five = six = 0
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    five += int(syndrome in pairs)
                    old = triples.get(syndrome, 0)
                    six += old
                    triples[syndrome] = old + 1
        return five // 10, six // 10

    def sampled_words(cols, side):
        """Actual kernel witnesses; minima remain only upper bounds.

        A common deterministic permutation stream compares finalists.
        Multiplicity is sampled, not a claimed logical spectrum.
        """
        srng = random.Random(491909 + 65537 * idx + (side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 12 and len(short) < 8192:
                short.add(word)

        for trial in range(64 if side == "A" else 32):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    record(word)
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for j in range(i):
                        candidate = word ^ basis[j]
                        record(candidate)
                        w = candidate.bit_count()
                        if candidate and w < weight:
                            word, weight = candidate, w
                    basis[i] = word
            if minimum <= 6:
                break

        loads = [0] * (len(cols) // q)
        low10 = 0
        for word in short:
            low10 += int(word.bit_count() <= 10)
            for j in range(len(loads)):
                loads[j] += int(bool((word >> (j * q)) & mask))
        return minimum, low10, len(short), max(loads)

    def portfolio(side):
        support = sa if side == "A" else sb
        positions = [(i, j) for i, row in enumerate(support)
                     for j in range(len(row))]
        pool, seen = [], set()
        serial = 0

        def analyze(grid):
            rows, width = component(grid, side)
            four, six, parts = _dic_cycles(rows, width)
            energy = (80 * four + six / 8.0) / q + 40 * (parts - 1)
            return energy, rows, width, four, six, parts

        def retain(grid, result):
            nonlocal serial
            _, rows, width, four, six, parts = result
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            # These exact component ranks are preferences imposed before
            # the complete-code rank is measured below.
            if gf2_rank(rows) != len(rows):
                return
            if side == "A":
                if gf2_rank([row & mask for row in rows]) != q:
                    return
            serial += 1
            key = (parts - 1, four, six, serial)
            pool.append((key, _dic_copy(grid), list(rows), width))
            pool.sort(key=lambda entry: entry[0])
            del pool[6:]

        for restart in range(4):
            grid = draw_grid(side)
            current = analyze(grid)
            retain(grid, current)
            for step in range(180 + 20 * (batch % 3)):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed = analyze(grid)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.982 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if step % 12 == 0:
                        retain(grid, current)
                else:
                    grid[i][j] = old
            retain(grid, current)

        if not pool:
            for _ in range(64):
                grid = draw_grid(side)
                retain(grid, analyze(grid))
                if pool:
                    break
        if not pool:
            raise ValueError("contract missed full-rank components")

        finalists = []
        for coarse, grid, rows, width in pool:
            # Confirm that cached singleton composition agrees exactly
            # with the scaffold's complete component probe.
            verified, verified_width = _dic_component(G, side, grid)
            assert rows == verified and width == verified_width
            cols = columns(rows, width)
            bad = int(_dic_small_kernel(rows, width))
            five, six = ((a_spectrum(cols) if not bad else (10**9, 10**9))
                         if side == "A" else (0, 0))
            d, low10, low12, peak = sampled_words(cols, side)
            boundary_peak, boundary_mean = boundary(cols)
            key = (bad, coarse[0], five + six, max(0, 12 - d),
                   low10, low12, peak, boundary_peak, boundary_mean,
                   coarse[1], coarse[2])
            finalists.append(dict(
                grid=grid, key=key, bad=bad, parts=coarse[0],
                five=five, six=six, d=d, low10=low10, low12=low12,
                peak=peak, boundary_peak=boundary_peak,
                boundary_mean=boundary_mean,
                four=coarse[1], cycle6=coarse[2]))
        finalists.sort(key=lambda entry: entry["key"])
        return finalists

    apool, bpool = portfolio("A"), portfolio("B")
    options = []
    for a in apool:
        for b in bpool:
            # Component words are replicated across five B columns or
            # two A columns. These scaled sampled exposures are only a
            # comparative surrogate, not a full quantum-logical census.
            a10, b10 = 5 * a["low10"], 2 * b["low10"]
            a12, b12 = 5 * a["low12"], 2 * b["low12"]
            key = (
                a["bad"] + b["bad"], a["parts"] + b["parts"],
                a["five"] + a["six"],
                max(0, 12 - a["d"]) + max(0, 12 - b["d"]),
                max(a10, b10), a10 + b10,
                max(a12, b12), a12 + b12,
                max(5 * a["peak"], 2 * b["peak"]),
                max(a["boundary_peak"], b["boundary_peak"]),
                a["boundary_mean"] + b["boundary_mean"],
                a["four"] + b["four"], a["cycle6"] + b["cycle6"],
            )
            options.append((key, a, b))
    options.sort(key=lambda entry: entry[0])

    verified_options = []
    for key, a, b in options:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or k != 144:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
        complete_key = key + (max(cx / len(hx), cz / len(hz)), cx + cz)
        verified_options.append((complete_key, A, B, hx, hz))
        if len(verified_options) == 6:
            break
    verified_options.sort(key=lambda entry: entry[0])

    fallback = None
    for key, A, B, hx, hz in verified_options:
        candidate = {
            "idx": idx, "case": case, "k": 144, "screen": key,
            "mixed_sites": tuple(sorted(mixed)),
            "cells": {"A": A, "B": B},
        }
        if fallback is None:
            fallback = candidate
        if _design_small_logical(hx, hz, 576):
            continue
        if _design_small_logical(hz, hx, 576):
            continue
        return candidate
    if fallback is not None:
        return fallback
    raise ValueError("no exact-bound unit/mixed-ratio pair passed rank screen")


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected D14 x C3 cell without late redraws or repair."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Higher k at equal d2s necessarily improves circuit score: current-parent slot10 had k=150,d2s=10,s=5.748, versus slot25 k=144,d2s=10,s=6.280; slot10 also required 29952 rather than 27936 CNOTs.
# REFUTED: Equal n,k,w,d2s and ASC depth fix circuit reliability: Dic12 slots4/8 both had n576,k144,w11,d2s12,depth14, but s=7.005/6.020; slot4 failed 4 Z and 9 X of 200000, while slot8 failed 101 Z of 148750 and 57 X of 200000.
# REFUTED: Unit-A mixed-ratio multiplicity screening guarantees zero-failure memory: all five measured current Dic12 codes still failed in both bases, with 10..21 Z and 15..31 X failures per 200000 shots.
# REFUTED: Exact rank-bound equality and matched screen/circuit summary statistics ensure comparable reliability on Frob39: slots35/45 both had n546,k156,d2s10,w12,depth15, but s=6.565/5.774 and Z failures33/200000 versus100/77250.
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
