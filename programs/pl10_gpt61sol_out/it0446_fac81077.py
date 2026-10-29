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
CONCEPT = ("C13 semidirect C3, action 3, order 39, shapes (1,2)x(2,6): joint-rank-safe mixed trinomials/quartets and balanced B columns should improve cell-0.300 circuit fitness beyond 6.0942 when selected by six-round decoder stress rather than distance alone.")
GROUP = ("Frozen L1: retain the exact C13:C3 group, order 39 and encoding. This bed already supports slot0's worse-basis s=6.493, although measured siblings scored 5.834..6.279. These readings motivate a filling experiment, not a causal group advantage.")
SHAPES = ("Frozen (1,2)x(2,6) targets cell0.300. Size factor14 allows q<=42; fixed q=39 gives n=546, kb=156 and guaranteed rate2/7. Unequal components have 78 variables/39 checks and 234 variables/78 checks. Parent failures diffusely enriched Acol/Bcol data and Acol/Brow checks around1.0-1.3x, collectively touching nearly all logicals; no stable block hotspot or transferable logical-qubit index is assumed.")
FILLING = ("Four architectures cross five indexed A component classes and B order schedules; larger slot budgets consume further classes. Joint component rank replaces individual A-cell invertibility. A counts are (3,3), (3,3), (3,4), (4,4); B row totals are9,9,8,8 respectively, with balanced columns of degrees2..4 and only singleton/doubleton cells. Maximum CSS weight remains12. Six-round CNOT budgets are23868,23868,24336,26208: stronger A protection is an explicit gate-cost experiment, not a gate-reduction claim. Scaffold-derived actions screen connectivity, component kernels, complete ranks, CSS and logicals through four. Common-stream six-round noisy-syndrome BP probes and independent holdouts select finalists. Probes omit ASC ordering and relay/LSD and are not calibrated circuit pL. Target the parent's Z failure caps, widespread collective flips, late settlement and unit-palette voids rather than maximize d2s.")
FAMILY_NAME = "r030_C13byC3_q39_12x26_joint_rank_temporal_holdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 semidirect C3 requires order 39")
    # x^13=y^3=1, y*x*y^-1=x^3; elements are x^a*y^b.
    # 3^3=1 mod 13 makes the C3 action well defined.
    m, r, s = 13, 3, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def idx(g):
        a, b = red(g)
        return a + m*b

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m)*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m)*a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _af_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _mc_brief(rng, G, idx, sa, sb):
    """Five algebraic contracts per support topology, not stored orbit copies."""
    from itertools import combinations

    case, batch = idx % 20, idx // 20
    model = case % 5
    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = order
                break
        else:
            raise ValueError("element order missing")

    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (3, 13)
    }
    triangles = {0: [], 1: []}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        spectrum = tuple(sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(G.mul(G.inv(u), v))],
        )))
        kind = {(3, 3, 13): 0, (3, 3, 3): 1}.get(spectrum)
        if kind is not None:
            triangles[kind].append([G.e, u, v])
    assert all(palettes.values()) and all(triangles.values())

    # A's two cells either both have one normal-subgroup edge,
    # have different spectra, or both span all quotient cosets.
    a_types = ((0, 0), (0, 0), (0, 1), (0, 1), (1, 1))[model]
    if batch % 2:
        a_types = tuple(reversed(a_types))
    kinds = {"A": {(0, j): a_types[j] for j in range(2)}, "B": {}}
    for i, row in enumerate(sb):
        for j, weight in enumerate(row):
            if weight == 3:
                kinds["B"][i, j] = (model + i + j + batch) % 2

    double_positions = [
        (i, j) for i, row in enumerate(sb)
        for j, weight in enumerate(row) if weight == 2
    ]
    # Exact numbers of order-three doubletons distinguish the briefs.
    # Extra harness batches extend the order schedule, not saved labels.
    wanted = (0, 1, 0, 2, 3)[model]
    wanted = min(len(double_positions), wanted + batch % 3)
    schedule = list(double_positions)
    rng.shuffle(schedule)
    short_positions = set(schedule[:wanted])
    doubles = {
        position: 3 if position in short_positions else 13
        for position in double_positions
    }
    return {
        "palettes": palettes, "triangles": triangles, "kinds": kinds,
        "doubles": doubles, "units": {}, "model": model,
        "order_three_doubletons": wanted,
    }


def _af_cell(rng, G, side, i, j, count, policy, batch):
    """Fresh cells satisfying this slot's noncommuting unit/order contract."""
    if count == 0:
        return []
    anchor = G.pick(rng)
    if count == 1:
        return [anchor]
    if count == 2:
        order = policy["doubles"][i, j]
        relative = rng.choice(policy["palettes"][order])
        return [anchor, G.mul(relative, anchor)]
    if count != 3:
        raise ValueError("unsupported metacyclic cell weight")

    kind = policy["kinds"][side][i, j]
    palette = policy["triangles"][kind]
    # A unit cell gives full row rank to A. Unit isolated B trinomials
    # also avoid a kernel confined to one lifted column block.
    # Rank is computed only from the scaffold's verified action.
    for _ in range(256):
        template = rng.choice(palette)
        signature = tuple(sorted(G.idx(g) for g in template))
        invertible = policy["units"].get(signature)
        if invertible is None:
            probe = lifted_matrices(G, [[template]], [[[]]])[0]
            invertible = gf2_rank(probe) == G.q
            policy["units"][signature] = invertible
        if not invertible:
            continue
        cell = [G.mul(g, anchor) for g in template]
        assert len({G._key(g) for g in cell}) == count
        return cell
    raise ValueError("unit-triangle palette search exhausted")


def _af_grid(rng, G, side, support, policy, batch):
    return [[_af_cell(rng, G, side, i, j, count, policy, batch)
             for j, count in enumerate(row)]
            for i, row in enumerate(support)]


def _af_rows(G, side, grid):
    """Obtain the component solely through the verified construction."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _af_cycles(rows, width):
    """Four-cycles, a six-cycle proxy, and check-graph components.

    Triangle-minus-stars is exact for six-cycles when four-cycles
    are absent; otherwise use it only as an optimization proxy.
    """
    size = len(rows)
    adjacency = [0] * size
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
            four += overlap*(overlap-1)//2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    triangles = 0
    for i in range(size):
        later = adjacency[i] >> (i+1)
        while later:
            bit = later & -later
            j = i+1+bit.bit_length()-1
            triangles += (
                (adjacency[i] & adjacency[j]) >> (j+1)
            ).bit_count()
            later ^= bit
    stars = sum(d*(d-1)*(d-2)//6 for d in degrees)

    unseen = (1 << size)-1
    parts = 0
    while unseen:
        todo = unseen & -unseen
        unseen ^= todo
        parts += 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            neighbors = adjacency[bit.bit_length()-1] & unseen
            unseen ^= neighbors
            todo |= neighbors
    return four, max(0, triangles-stars), parts


def _af_small_kernel(rows, width, limit):
    """Exact kernel existence test through weight four or six."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= 1 << r
            row ^= bit
    seen = {0}
    for column in columns:
        if column in seen:
            return True
        seen.add(column)
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            if syndrome in seen:
                return True
            seen.add(syndrome)
    if limit == 6:
        for i, column in enumerate(columns):
            for j in range(i):
                pair = column ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True
                    seen.add(syndrome)
    return False


def _af_kernel_scan(rng, rows, width, trials):
    """Sample actual kernel words and short-word multiplicity.

    The minimum is an upper bound, never a distance certificate or
    a circuit-failure estimate. All sampled words use verified rows.
    """
    minimum = width
    small = set()

    def inspect(word):
        weight = word.bit_count()
        if weight <= 12:
            small.add(word)
        return weight

    for _ in range(trials):
        mat = list(rows)
        order = list(range(width))
        rng.shuffle(order)
        pivots = []
        rank = 0
        for c in order:
            bit = 1 << c
            found = next((r for r in range(rank, len(mat))
                          if mat[r] & bit), None)
            if found is None:
                continue
            mat[rank], mat[found] = mat[found], mat[rank]
            pivot = mat[rank]
            for r in range(len(mat)):
                if r != rank and mat[r] & bit:
                    mat[r] ^= pivot
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
            minimum = min(minimum, inspect(word))
        for i, word in enumerate(basis):
            for j in range(i):
                minimum = min(minimum, inspect(word ^ basis[j]))
        if len(basis) >= 3:
            for shot in range(256):
                a, b, c = rng.sample(range(len(basis)), 3)
                minimum = min(
                    minimum, inspect(basis[a] ^ basis[b] ^ basis[c]))
        if minimum <= 4:
            break
    return minimum, len(small)


def _af_select(rng, G, side, support, policy, batch):
    """Search a fixed count/order contract, then screen decoder-risk surrogates."""
    import math

    positions = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
    pool, identities = [], set()
    serial = 0

    def analyze(grid):
        rows, width = _af_rows(G, side, grid)
        four, six, parts = _af_cycles(rows, width)
        deficit = len(rows) - gf2_rank(rows)
        energy = ((100*four + six/6.0)/G.q
                  + 40*(parts-1) + 20*deficit)
        return energy, rows, width, four, six, parts, deficit

    def retain(grid, result):
        nonlocal serial
        _, rows, width, four, six, parts, deficit = result
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in identities:
            return
        identities.add(signature)
        serial += 1
        key = (parts-1, deficit, four, six, serial)
        pool.append((key, _af_copy(grid), list(rows), width))
        pool.sort(key=lambda item: item[0])
        del pool[10:]

    for restart in range(4):
        grid = _af_grid(rng, G, side, support, policy, batch)
        current = analyze(grid)
        retain(grid, current)
        local_grid, local = _af_copy(grid), current
        for step in range(220):
            i, j = rng.choice(positions)
            previous = grid[i][j]
            grid[i][j] = _af_cell(
                rng, G, side, i, j, support[i][j], policy, batch)
            proposed = analyze(grid)
            delta = proposed[0] - current[0]
            temperature = 18.0*(1.0-step/220.0)**2 + 0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if (current[0] < local[0] or
                        (current[0] == local[0] and rng.random() < 0.08)):
                    local_grid, local = _af_copy(grid), current
            else:
                grid[i][j] = previous
            if step in (109, 219):
                retain(grid, current)
        retain(local_grid, local)

    def boundary_exposure(rows, width):
        """Small-boundary triples from the verified component.

        These are ambiguity/exposure surrogates, not measured absorbing
        sets, ASC hooks, or a circuit-level failure prediction.
        """
        columns = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length()-1] |= marker
                row ^= bit
        total = 0
        loads = [0] * (width // G.q)
        for i, ci in enumerate(columns):
            for j in range(i):
                pair = ci ^ columns[j]
                for h in range(j):
                    boundary = (pair ^ columns[h]).bit_count()
                    if boundary <= 3:
                        cost = (5-boundary)**2
                        total += cost
                        loads[i // G.q] += cost
                        loads[j // G.q] += cost
                        loads[h // G.q] += cost
        return total / width, max(loads) / G.q

    finalists = []
    for cost, grid, rows, width in pool[:5]:
        # Unlike the parent, test both components through weight six.
        bad = _af_small_kernel(rows, width, 6)
        distance, multiplicity = _af_kernel_scan(
            rng, rows, width, 40 if side == "A" else 24)
        risk, peak = boundary_exposure(rows, width)
        target = 12 if side == "A" else 10
        key = (
            cost[0], int(bad or distance <= 6), cost[1],
            max(0, target-distance), cost[2],
            multiplicity, peak, risk, cost[3], cost[4],
        )
        finalists.append({
            "grid": grid, "cost": cost, "key": key,
            "bad": int(bad or distance <= 6),
            "distance": distance, "multiplicity": multiplicity,
            "risk": risk, "peak": peak,
        })
    finalists.sort(key=lambda item: item["key"])
    if not finalists:
        raise ValueError("component search produced no finalist")
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Indexed joint-rank designs and six-round decoder holdouts."""
    import math
    import hashlib
    from collections import Counter
    from itertools import combinations
    import numpy as np

    q = G.q
    assert q == 39 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    architecture, number = idx % 4, idx // 4
    a_counts = ((3, 3), (3, 3), (3, 4), (4, 4))[architecture]
    profiles = (
        ((1,2), (1,2), (1,2), (2,1), (2,1), (2,1)),
        ((1,1), (1,2), (1,2), (2,1), (2,1), (2,2)),
        ((1,1), (1,1), (1,2), (1,2), (2,1), (2,1)),
        ((1,1), (1,1), (1,2), (1,2), (2,1), (2,1)),
    )
    profile = profiles[architecture]
    sa = [list(a_counts)]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    expected_cnots = (23868, 23868, 24336, 26208)[architecture]

    # Cache only singleton actions obtained from the verified helper.
    mask = (1 << q)-1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for cell_row in grid:
            out = [0]*q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for r, word in enumerate(actions[side][G.idx(g)]):
                        out[r] ^= word << (j*q)
            rows.extend(out)
        return rows

    def columns(rows, width):
        out = [0]*width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def fingerprint(rows, width):
        """Rooted uncolored invariant, not full-code canonization.

        The one-row A check orbit is transitive. Unequal fingerprints
        distinguish its component graphs; the evaluator remains the
        authority for arbitrary equivalence of complete LP codes.
        """
        height = len(rows)
        adjacency = [[] for _ in range(height+width)]
        for c, syndrome in enumerate(columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length()-1
                adjacency[r].append(height+c)
                adjacency[height+c].append(r)
                syndrome ^= bit
        initial = [(int(v >= height), len(ns), int(v == 0))
                   for v, ns in enumerate(adjacency)]
        palette = {key: t for t, key in enumerate(sorted(set(initial)))}
        colors = [palette[key] for key in initial]
        digest = hashlib.sha256()
        for _ in range(10):
            signatures = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            digest.update(repr(sorted(Counter(signatures).items())).encode())
            palette = {key: t for t, key in enumerate(
                sorted(set(signatures)))}
            colors = [palette[key] for key in signatures]
        return digest.hexdigest()

    def sampled_words(rows, width, seed, trials):
        """Genuine component witnesses; minima are only upper bounds."""
        source = random.Random(seed)
        cs = columns(rows, width)
        order = list(range(width))
        minimum, short = width+1, set()
        for _ in range(trials):
            source.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cs[c], 1 << c
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
            basis.sort(key=int.bit_count)
            reducers = basis[:32]
            for word in basis:
                for _ in range(3):
                    candidate = min(
                        [word]+[word ^ other for other in reducers
                                if word != other],
                        key=int.bit_count)
                    if candidate == word:
                        break
                    word = candidate
                if word:
                    weight = word.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 10:
                        short.add(word)
            if minimum <= 4:
                break
        return minimum, len(short)

    # An all-coset trinomial need not be a unit. Rejecting its entire
    # palette was the parent's slot-loss mechanism. Use joint rank.
    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[G.idx(g)] = order
                break
    triangle_books = {"edge": [], "coset": []}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        cell = [G.e, u, v]
        phases = {g[1] for g in cell}
        kind = "coset" if len(phases) == 3 else "edge"
        triangle_books[kind].append(cell)
    assert all(triangle_books.values())

    def draw_A(source):
        result = []
        for j, count in enumerate(a_counts):
            if count == 3:
                kind = ("edge" if architecture < 2 and j == 0
                        else "coset")
                result.append(list(source.choice(triangle_books[kind])))
            else:
                while True:
                    cell = [G.e]+source.sample(nonidentity, 3)
                    if any(G.mul(u, v) != G.mul(v, u)
                           for u, v in combinations(cell[1:], 2)):
                        result.append(cell)
                        break
        return [result]

    # Every call rebuilds the SAME architecture catalogue. Slot number
    # chooses a component class, not a relabeling or another random draw.
    source = random.Random(391326041+1009*architecture)
    needed = max(5, (N_SLOTS+3)//4)
    proposals, labels = [], set()
    target_rank = q-1 if architecture == 3 else q
    for serial in range(max(600, 64*needed)):
        A = draw_A(source)
        label = tuple(tuple(sorted(G.idx(g) for g in cell))
                      for cell in A[0])
        if label in labels:
            continue
        labels.add(label)
        ar = component(A, "A")
        four, six, parts = _af_cycles(ar, 2*q)
        if parts != 1 or gf2_rank(ar) != target_rank:
            continue
        if _af_small_kernel(ar, 2*q, 4):
            continue
        proposals.append(((four, six, label), serial, A, ar))
    proposals.sort(key=lambda item: item[:2])

    bank, identities = [], set()
    for coarse, serial, A, ar in proposals:
        identity = fingerprint(ar, 2*q)
        if identity in identities:
            continue
        identities.add(identity)
        bad6 = int(_af_small_kernel(ar, 2*q, 6))
        bank.append(((bad6, coarse[:2]), identity, serial, A, ar))
        if len(bank) >= needed+8:
            break
    bank.sort(key=lambda item: (item[0], item[1]))
    if len(bank) <= number:
        # Never transform a saved code to fill a missing class. This
        # fallback changes the actual relative supports and B contract;
        # downstream full Tanner-isomorphism testing remains necessary.
        if len(proposals) <= number:
            raise ValueError("joint-rank A catalogue exhausted")
        coarse, serial, A, ar = proposals[number]
        a_identity = fingerprint(ar, 2*q)
        abad = int(_af_small_kernel(ar, 2*q, 6))
    else:
        (abad, _), a_identity, serial, A, ar = bank[number]
    da, na = sampled_words(ar, 2*q, 391001+idx, 40)

    doubles = [(i, j) for i in range(2) for j in range(6)
               if sb[i][j] == 2]
    # Indexed order schedules are part of the experiment, while the
    # indexed A graph supplies the main relative-geometry distinction.
    short_count = min(len(doubles), number % 5)
    schedule_rng = random.Random(391777+7919*idx)
    short_sites = set(schedule_rng.sample(doubles, short_count))
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (3, 13)
    }

    def draw_B_cell(i, j):
        anchor = G.pick(rng)
        if sb[i][j] == 1:
            return [anchor]
        order = 3 if (i, j) in short_sites else 13
        delta = rng.choice(palettes[order])
        return [anchor, G.mul(anchor, delta)]

    def draw_B():
        return [[draw_B_cell(i, j) for j in range(6)] for i in range(2)]

    def assess(B):
        br = component(B, "B")
        four, six, parts = _af_cycles(br, 6*q)
        deficit = 2*q-gf2_rank(br)
        energy = (80*deficit+50*(parts-1)
                  +(100*four+six/8.0)/q)
        return energy, (parts-1, deficit, four, six), br

    def temporal_probe(checks, stabilizers, seed, shots):
        """Six-round phenomenological BP, NOT the scored ASC experiment.

        Includes noisy syndrome histories and approximate spread pairs.
        Exposure rates, min-sum BP and the perfect closing syndrome are
        surrogate choices; ASC gate ordering and relay/LSD are absent.
        """
        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length()-1)
                row ^= bit
            return out

        n, rounds, m = 546, 6, len(checks)
        ns = [sites(row) for row in checks]
        hooks = [sites(row) for row in stabilizers]
        variables = rounds*(n+m)
        detector_rows = []
        for t in range(rounds+1):
            for r, row in enumerate(ns):
                rr = [t*n+c for c in row] if t < rounds else []
                if t < rounds:
                    rr.append(rounds*n+t*m+r)
                if t:
                    rr.append(rounds*n+(t-1)*m+r)
                detector_rows.append(rr)

        ne = sum(map(len, detector_rows))
        maximum = max(map(len, detector_rows))
        ce = np.full((len(detector_rows), maximum), ne, dtype=np.int32)
        cv = np.full((len(detector_rows), maximum),
                     variables, dtype=np.int32)
        valid = np.zeros(ce.shape, dtype=bool)
        ev, incident = [], [[] for _ in range(variables)]
        for r, row in enumerate(detector_rows):
            start = len(ev)
            ce[r, :len(row)] = np.arange(start, start+len(row))
            cv[r, :len(row)] = row
            valid[r, :len(row)] = True
            for c in row:
                incident[c].append(len(ev))
                ev.append(c)
        ve = np.full((variables, max(map(len, incident))),
                     ne, dtype=np.int32)
        for c, edges in enumerate(incident):
            ve[c, :len(edges)] = edges
        ev = np.asarray(ev, dtype=np.int32)

        gate_degree = np.asarray([
            x.bit_count()+z.bit_count()
            for x, z in zip(columns(checks, n),
                            columns(stabilizers, n))], dtype=float)
        data_rate = np.minimum(0.002*(1+1.5*gate_degree), 0.12)
        measure_rate = np.asarray([
            0.002*(2+len(row))*2/3 for row in ns])
        base = np.concatenate((np.tile(data_rate, rounds),
                               np.tile(measure_rate, rounds)))
        levels = np.asarray((0.50, 0.70, 0.90))[np.arange(shots) % 3]
        rates = np.minimum(base[None, :]*levels[:, None], 0.25)
        channel = np.random.default_rng(seed)
        errors = np.zeros((shots, variables+1), dtype=bool)
        errors[:, :variables] = (
            channel.random((shots, variables)) < rates)
        for t in range(rounds):
            chosen = channel.integers(len(hooks), size=shots)
            active = channel.random(shots) < 0.25
            for shot in np.flatnonzero(active):
                row = hooks[int(chosen[shot])]
                if len(row) >= 2:
                    a, b = channel.choice(len(row), 2, replace=False)
                    errors[shot, t*n+row[int(a)]] ^= True
                    errors[shot, t*n+row[int(b)]] ^= True

        target = np.logical_xor.reduce(
            errors[:, cv] & valid[None, :, :], axis=2)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.log((1-rates)/rates)
        messages = prior[:, ev].copy()
        outgoing = np.zeros((shots, ne+1), dtype=float)
        truth = np.logical_xor.reduce(
            errors[:, :rounds*n].reshape(shots, rounds, n), axis=1)
        done = np.zeros(shots, dtype=bool)
        decoded = np.zeros((shots, n), dtype=bool)
        completion = np.full(shots, 33, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]

        for iteration in range(1, 33):
            padded = np.concatenate(
                (messages, np.full((shots, 1), 1.0e6)), axis=1)
            incoming = padded[:, ce]
            magnitude = np.abs(incoming)
            first_at = magnitude.argmin(axis=2)
            first = magnitude.min(axis=2)
            second = np.partition(magnitude, 1, axis=2)[:, :, 1]
            signs = np.where(incoming < 0, -1.0, 1.0)
            product = signs.prod(axis=2)*target_sign
            excluded = np.where(
                positions == first_at[:, :, None],
                second[:, :, None], first[:, :, None])
            update = 0.8*product[:, :, None]*signs*excluded
            outgoing[:, :ne] = update[:, valid]
            posterior = prior+outgoing[:, ve].sum(axis=2)
            hard = posterior < 0
            padded_hard = np.concatenate(
                (hard, np.zeros((shots, 1), dtype=bool)), axis=1)
            syndrome = np.logical_xor.reduce(
                padded_hard[:, cv] & valid[None, :, :], axis=2)
            newly = ~np.any(syndrome != target, axis=1) & ~done
            decoded[newly] = np.logical_xor.reduce(
                hard[newly, :rounds*n].reshape(-1, rounds, n), axis=1)
            completion[newly] = iteration
            done |= newly
            if done.all():
                break
            messages = np.clip(
                0.30*messages+0.70*(posterior[:, ev]-outgoing[:, :ne]),
                -30, 30)

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        wrong = 0
        for shot in np.flatnonzero(done):
            word = sum(1 << int(c) for c in np.flatnonzero(
                decoded[shot] ^ truth[shot]))
            while word:
                p = word.bit_length()-1
                if p not in pivots:
                    wrong += 1
                    break
                word ^= pivots[p]
        unresolved = int((~done).sum())
        return wrong+unresolved, unresolved, wrong, int(completion.sum())

    def quality(x, z):
        return tuple(v for a, b in zip(x, z) for v in (max(a, b), a+b))

    fallback = None
    for batch in range(2):
        pool, encountered = [], set()
        counter = 0

        def retain(B, result):
            nonlocal counter
            _, coarse, br = result
            label = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
            if label in encountered:
                return
            encountered.add(label)
            counter += 1
            pool.append((coarse, counter, _af_copy(B), list(br)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(3):
            B = draw_B()
            current = assess(B)
            retain(B, current)
            for step in range(160):
                i, j = rng.randrange(2), rng.randrange(6)
                old = B[i][j]
                B[i][j] = draw_B_cell(i, j)
                proposed = assess(B)
                delta = proposed[0]-current[0]
                temperature = 12.0*(0.978**step)+0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                    if delta < 0 or step % 24 == 0:
                        retain(B, current)
                else:
                    B[i][j] = old
            retain(B, current)

        options = []
        for coarse, serial_b, B, br in pool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 546 or not 275*n <= 1000*k < 325*n:
                continue
            assert css_ok(hx, hz)
            assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
            assert max(row.bit_count() for row in hx+hz) <= W_MAX

            # Independently confirm cached components against the full lift.
            actual_a = [
                sum(((hx[r] >> (j*6*q)) & mask) << (j*q)
                    for j in range(2)) for r in range(q)]
            actual_b = [row & ((1 << (6*q))-1) for row in hz[:2*q]]
            assert actual_a == ar and actual_b == br
            bad4 = int(_af_small_kernel(br, 6*q, 4))
            db, nb = sampled_words(br, 6*q, 391002+idx, 20)
            logical_bad = int(
                _design_small_logical(hx, hz, n)
                or _design_small_logical(hz, hx, n))
            protection = (logical_bad, max(abad, bad4), coarse[0],
                          max(0, 12-da)+max(0, 10-db))
            tie = (coarse[1:], max(na, nb), na+nb, serial_b)
            item = (protection, tie, B, hx, hz, k, db)
            options.append(item)
        options.sort(key=lambda item: item[:2])
        if options and fallback is None:
            fallback = options[0]

        seed = 391546031+1009*idx
        coarse_probes = []
        for item in options[:4]:
            protection, tie, B, hx, hz, k, db = item
            x = temporal_probe(hz, hx, seed, 18)
            z = temporal_probe(hx, hz, seed+1, 18)
            # Exact short-logical warnings precede the probe. Sampled
            # distance shortfalls follow it, rather than masquerade as pL.
            key = protection[:3]+quality(x, z)+(protection[3], tie)
            coarse_probes.append((key, item, x, z))
        coarse_probes.sort(key=lambda item: item[0])

        winner = None
        for _, item, cx, cz in coarse_probes[:2]:
            protection, tie, B, hx, hz, k, db = item
            x = temporal_probe(hz, hx, seed+1000003, 48)
            z = temporal_probe(hx, hz, seed+1000004, 48)
            tx = tuple(a+b for a, b in zip(cx, x))
            tz = tuple(a+b for a, b in zip(cz, z))
            key = protection[:3]+quality(tx, tz)+(protection[3], tie)
            if winner is None or key < winner[0]:
                winner = key, item, tx, tz
        if winner is not None:
            key, item, tx, tz = winner
            break
    else:
        if fallback is None:
            for attempt in range(96):
                B = draw_B()
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n == 546 and 275*n <= 1000*k < 325*n:
                    fallback = ((1, 1, 1, 1000), (attempt,),
                                B, hx, hz, k, 0)
                    break
        if fallback is None:
            raise ValueError("no joint-rank filling retained cell0.300")
        item = fallback
        key, tx, tz = (10**12,), None, None

    _, _, B, hx, hz, k, db = item
    assert css_ok(hx, hz) and lifted_k(G, A, B) == k
    assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    assert all(len(cell) == len(set(cell)) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    return {
        "idx": idx, "architecture": architecture, "A_class": a_identity,
        "A_number": number, "k": k, "cnots": expected_cnots,
        "order_three_doubletons": short_count,
        "component_upper_bounds": (da, db),
        "temporal_bp_surrogate": (tx, tz),
        "screen": key, "cells": {"A": A, "B": B},
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without redraws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Low ASC depth alone guarantees basis-balanced reliability: current-parent slot42 had depth14 in both bases but 11/200000 Z failures versus 91/200000 X failures.
# REFUTED: Equal d2s, row weight and ASC depth certify comparable circuit reliability: current slots4/17 both had d2s=10, w=12 and depth15, but Z failed 100/169750 versus 102/13750 shots, with worse-basis s=6.148 versus 5.048.
# REFUTED: Individual-unit triangle contracts reliably populate this bed and suppress collective failures: only12/60 slots were distinct,48 exhausted the unit palette, and measured d2s=10 siblings ranged from45/200000 to100/88750 Z failures while flipping all156 logicals.
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
