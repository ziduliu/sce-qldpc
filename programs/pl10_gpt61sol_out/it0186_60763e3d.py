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
CONCEPT = ("For the binary octahedral group 2O, order 48, shapes A=(1,2), B=(2,5), quaternion-unit trinomials and shared-noise decoder-ambiguity screening should raise cell 0.250's mean worse-basis circuit fitness above the GL(2,3) parent's 6.7781.")
GROUP = ("Jump to 2O, represented by exact unit quaternions over Q(sqrt(2)). It has one nonidentity involution, versus thirteen in GL(2,3), so the jump is not a relabeling. Normal Q8 preserves odd-augmentation and quotient-monomial unit constructions. Order-four elements outside Q8 supply additional connection directions. The measured GL(2,3) and Dic12 order-48 (1,2)x(2,5) beds both reached about 6.78; these motivate retaining q=48, not a claimed group advantage.")
SHAPES = ("Retain (1,2)x(2,5) for cell 0.250: n=576, kb=144 and guaranteed rate 0.250. The size factor permits q<=49, and this new group fixes q=48. A has 96 variables/48 checks and B has 240 variables/96 checks; final verified ranks require k=144. Parent slot3's X map implicated Acol0/Bcol2,3 at 1.3x, whereas slot6's failures spread across other B columns. Mostly weak Acol/Brow check enrichment does not establish a persistent local repair target.")
FILLING = ("Twenty contracts retain A weights (3,3)/(3,4), B totals 15/16 and five relative-order portfolios. A0 uses normal-Q8 unit trinomials; the partner and B use actual 2O elements of orders four, six and eight. HX weights are <=11 and HZ weights <=12, with 25056-27936 six-round CNOTs. Helper-derived component actions guide annealing and short-kernel screening. Above the capped protection floor, optional NumPy normalized min-sum probes compare finalists on common independent p=.025 errors and fixed weights 4..7; replication-weighted decoding ambiguity precedes cycle tie-breakers. This is an uncalibrated classical surrogate, not relay-BP or circuit pL. Parent slots3/6 had d2s12 and depth14 but 155/205 X LSD resolutions and 8/34 X failures, motivating decoder-aware selection. Final helper-built matrices enforce k144, CSS, weight legality and prefer no logical <=4.")
FAMILY_NAME = "r025_binaryO48_12x25_quaternion_unit_sharedBP_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("binary octahedral 2O requires order 48")
    from itertools import product, combinations

    # Quaternion coordinate t is (g[2*t]+g[2*t+1]*sqrt(2))/2.
    # 8 coordinate units, 16 Hurwitz half-units, 24 two-axis units.
    els = []
    for axis in range(4):
        for sign in (-1, 1):
            g = [0] * 8
            g[2*axis] = 2*sign
            els.append(tuple(g))
    for signs in product((-1, 1), repeat=4):
        els.append(tuple(v for sign in signs for v in (sign, 0)))
    for a, b in combinations(range(4), 2):
        for sa, sb in product((-1, 1), repeat=2):
            g = [0] * 8
            g[2*a+1], g[2*b+1] = sa, sb
            els.append(tuple(g))
    positions = {g: i for i, g in enumerate(els)}
    ident = (2, 0, 0, 0, 0, 0, 0, 0)
    assert len(positions) == 48

    def red(g):
        return tuple(g)

    # Signed quaternion-unit multiplication; abs(entry)-1 is the axis.
    units = ((1, 2, 3, 4), (2, -1, 4, -3),
             (3, -4, -1, 2), (4, 3, -2, -1))

    def hamilton(g, h):
        out = [0] * 8
        for i in range(4):
            a, b = g[2*i], g[2*i+1]
            for j in range(4):
                c, d = h[2*j], h[2*j+1]
                entry = units[i][j]
                axis = 2*(abs(entry)-1)
                sign = 1 if entry > 0 else -1
                out[axis] += sign*(a*c + 2*b*d)
                out[axis+1] += sign*(a*d + b*c)
        assert all(v % 2 == 0 for v in out)
        return tuple(v // 2 for v in out)

    # Immutable-in-use group data, not a cache of candidate fillings.
    table = {(g, h): hamilton(g, h) for g in els for h in els}
    assert all(g in positions for g in table.values())

    def mul(g, h):
        return table[red(g), red(h)]

    def inv(g):
        g = red(g)
        return g[:2] + tuple(-v for v in g[2:])

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=ident, mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty group-algebra contracts, optimized without shared state."""
    import math
    from itertools import combinations

    design = idx % 20
    mode = (idx // 20) % 3
    architecture, order_design = divmod(design, 5)
    aw = (3, 3) if architecture < 2 else (3, 4)
    b_profiles = (
        ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
        ((1, 2), (1, 2), (2, 1), (2, 1), (2, 2)),
    )
    bp = architecture % 2
    sa = [list(aw)]
    sb = [[column[i] for column in b_profiles[bp]]
          for i in range(2)]
    assert _no_dead_lines(sa, sb)
    # Check the two budgets explicitly, not just total gate count.
    wx = sum(aw) + max(map(sum, b_profiles[bp]))
    wz = max(map(sum, sb)) + max(aw)
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)

    q = G.q
    widths = (A_SHAPE[1] * q, B_SHAPE[1] * q)
    shapes = (A_SHAPE, B_SHAPE)
    elements = list(G.elements)

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(G.e):
                return t
        raise ValueError("element order failed")

    orders = {G._key(g): element_order(g) for g in elements}
    by_order = {
        t: [g for g in elements if orders[G._key(g)] == t]
        for t in (4, 6, 8)
    }
    # Q8 consists of the eight rational coordinate units. The other
    # order-four elements of 2O have nonzero sqrt(2) coefficients.
    normal_four = [
        g for g in by_order[4]
        if all(g[2*t+1] == 0 for t in range(4))
    ]
    quaternion = [G.e, tuple(-v for v in G.e)] + normal_four
    assert len({G._key(g) for g in quaternion}) == 8
    assert all(
        G._key(G.mul(g, h)) in {G._key(x) for x in quaternion}
        for g in quaternion for h in quaternion
    )

    def relative_order(g, h):
        return orders[G._key(G.mul(G.inv(g), h))]

    def pair_book(first, second, final_order):
        return [(h, z) for h in first for z in second
                if relative_order(h, z) == final_order]

    if mode == 0:
        # Three terms in one Q8 coset have odd augmentation in F2[Q8].
        # Their unit property survives multiplication by a group element.
        unit_pairs = pair_book(normal_four, normal_four, 4)
    else:
        # The first two terms cancel modulo normal Q8. The remaining
        # monomial is a unit in the quotient; the kernel ideal is nilpotent.
        unit_pairs = pair_book(normal_four, by_order[6], 6)

    if mode == 2:
        partner_pairs = pair_book(by_order[8], by_order[8], 4)
    else:
        partner_pairs = pair_book(by_order[6], by_order[8], 8)

    # Four-term partners retain the prescribed triangle but add a new
    # element. Exclude involutory and order-three pair differences.
    partner_book = []
    for h, z in partner_pairs:
        extensions = tuple(
            w for w in elements
            if all(relative_order(u, w) in (4, 6, 8)
                   for u in (G.e, h, z))
        ) if aw[1] == 4 else ()
        if aw[1] == 3 or extensions:
            partner_book.append((h, z, extensions))
    assert unit_pairs and partner_book

    b_palettes = ((8,), (6,), (4,), (8, 4), (8, 6))
    b_palette = b_palettes[order_design]

    def draw_cell(side, i, j):
        if side == 0:
            g = G.pick(rng)
            if j == 0:
                h, z = rng.choice(unit_pairs)
                return [g, G.mul(g, h), G.mul(g, z)]
            h, z, extensions = rng.choice(partner_book)
            cell = [g, G.mul(g, h), G.mul(g, z)]
            if aw[j] == 4:
                cell.append(G.mul(g, rng.choice(extensions)))
            return cell
        g = G.pick(rng)
        if sb[i][j] == 1:
            return [g]
        order = b_palette[(j + 2*i) % len(b_palette)]
        return [g, G.mul(g, rng.choice(by_order[order]))]

    def labels(counts, side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton actions extracted from lifted_matrices.
    # Composing these blocks cannot introduce a different lift convention.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G._key(g)] = tuple(row & mask for row in hx)
        actions[1][G._key(g)] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def cycle_cost(rows):
        """Exact component 4/6-cycle counts and a rank-deficit penalty."""
        m = len(rows)
        width = max(row.bit_length() for row in rows)
        columns = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1].append(r)
                row ^= bit

        adj = [dict() for _ in range(m)]
        for ns in columns:
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1

        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        if len(seen) != m:
            return None

        four = sum(v * (v - 1) // 2
                   for a in range(m) for b, v in adj[a].items() if a < b)
        six = 0
        for a in range(m):
            for b, ab in adj[a].items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in adj[a]:
                        six += ab * bc * adj[a][c]
        # Remove triangles whose check-pair edges reuse a variable.
        for ns in columns:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        deficit = m - gf2_rank(rows)
        return (four, six + q * deficit, deficit)

    pools = [[], []]

    def energy(cost):
        four, six_with_deficit, deficit = cost
        return (100 * four + six_with_deficit + 4 * q * deficit) / q

    for side, counts in enumerate((sa, sb)):
        seen_fillings = set()

        def offer(cost, serial, grid, rows):
            signature = tuple(tuple(sorted(G._key(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen_fillings:
                return
            seen_fillings.add(signature)
            pools[side].append((cost, serial, clone(grid), list(rows)))
            pools[side].sort(key=lambda item: (item[0], item[1]))
            del pools[side][6:]

        # Optimize voltages inside the architecture, rather than merely
        # increasing the number of independent random draws.
        for restart in range(4):
            for attempt in range(16):
                grid = labels(counts, side)
                rows = component_rows(grid, side)
                current = cycle_cost(rows)
                if current is not None:
                    break
            else:
                continue

            best = current
            best_grid, best_rows = clone(grid), list(rows)
            offer(current, restart * 161, grid, rows)
            for step in range(160):
                nr, nc = shapes[side]
                i, j = rng.randrange(nr), rng.randrange(nc)
                old = grid[i][j]
                if side == 1 and rng.random() < 0.25:
                    # Preserve the internal order contract while changing
                    # inter-cell voltages on the two-row B component.
                    shift = G.pick(rng)
                    proposed_cell = [G.mul(shift, g) for g in old]
                else:
                    # Whole-cell translations on one-row A are gauges;
                    # redraw its internal geometry instead.
                    proposed_cell = draw_cell(side, i, j)
                assert len(proposed_cell) == len(old)
                assert len({G._key(g) for g in proposed_cell}) == len(old)
                grid[i][j] = proposed_cell

                proposed_rows = component_rows(grid, side)
                proposed = cycle_cost(proposed_rows)
                if proposed is None:
                    grid[i][j] = old
                    continue
                difference = energy(proposed) - energy(current)
                temperature = 24.0 * (0.975 ** step) + 0.15
                accept = (difference <= 0 or
                          rng.random() < math.exp(
                              -min(700.0, difference / temperature)))
                if not accept:
                    grid[i][j] = old
                    continue

                current, rows = proposed, proposed_rows
                improved = current < best
                if improved:
                    best = current
                    best_grid, best_rows = clone(grid), list(rows)
                if improved or step % 8 == 0:
                    offer(current, restart * 161 + step + 1, grid, rows)
            offer(best, 1000 + restart, best_grid, best_rows)

    if not all(pools):
        raise ValueError("no connected low-gate component candidate")

    def kernel_upper_bound(rows, n):
        """Sample kernel words using randomized information sets.

        This is only an upper-bound heuristic, not a distance certificate
        and not an estimate of the scored circuit-level failure rate.
        """
        best = n
        for _ in range(96):
            mat = list(rows)
            order = list(range(n))
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

            for i, x in enumerate(basis):
                for j in range(i):
                    best = min(best, (x ^ basis[j]).bit_count())
            if best <= 8:
                break
        return best

    def columns_of(rows, ncols):
        columns = [0] * ncols
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return columns

    def small_kernel_and_boundary(rows, width):
        # Equal syndromes of distinct subsets of size <=3 exactly
        # detect a nonzero component kernel word of weight <=6.
        columns = columns_of(rows, width)
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 0
            seen.add(value)
        for i, value in enumerate(columns):
            for j in range(i):
                syndrome = value ^ columns[j]
                if syndrome in seen:
                    return True, 0
                seen.add(syndrome)

        boundary = 0
        for i, value in enumerate(columns):
            for j in range(i):
                pair = value ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True, boundary
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight < 5:
                        boundary += (5 - weight) ** 2
        # The components now have different lengths. Compare boundary
        # exposure per variable rather than favoring the shorter side.
        return False, boundary / width

    def has_small_logical(hx, hz, n):
        # Exact logical-weight<=4 test on the full verified matrices.
        # One representative per syndrome suffices modulo stabilizers.
        for checks, stabilizers in ((hx, hz), (hz, hx)):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    if p in pivots:
                        row ^= pivots[p]
                    else:
                        pivots[p] = row
                        break

            def trivial(word):
                while word:
                    p = word.bit_length() - 1
                    if p not in pivots:
                        return False
                    word ^= pivots[p]
                return True

            columns = columns_of(checks, n)
            bits = [1 << i for i in range(n)]
            seen = {0: 0}
            for i, syndrome in enumerate(columns):
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = bits[i]
                elif not trivial(previous ^ bits[i]):
                    return True
            for i, syndrome_i in enumerate(columns):
                for j in range(i):
                    syndrome = syndrome_i ^ columns[j]
                    word = bits[i] ^ bits[j]
                    previous = seen.get(syndrome)
                    if previous is None:
                        seen[syndrome] = word
                    elif not trivial(previous ^ word):
                        return True
        return False

    try:
        import numpy as np
    except ImportError:
        np = None

    noise_books = {}
    if np is not None:
        # Independent of annealing draws; every finalist on a side gets
        # exactly the same errors. This is not ASC circuit simulation.
        for side, width in enumerate(widths):
            nrng = random.Random(840113 + 65537*idx + 101*side)
            errors = np.zeros((width, 256), dtype=np.uint8)
            for shot in range(160):
                for c in range(width):
                    errors[c, shot] = int(nrng.random() < 0.025)
            for shot in range(160, 256):
                weight = 4 + (shot-160) % 4
                errors[nrng.sample(range(width), weight), shot] = 1
            noise_books[side] = errors

    def decoder_probe(rows, side):
        """Classical normalized min-sum on a verified component.

        Counts include unsettled shots and wrong kernel representatives.
        This is neither stabilizer-aware relay-BP nor measured circuit pL.
        If NumPy is unavailable, structural screening remains operative.
        """
        if np is None:
            return (0, 0, 0)
        errors = noise_books[side]
        width, shots = errors.shape
        matrix = np.zeros((len(rows), width), dtype=np.uint8)
        variables = []
        row_edges = []
        var_edges = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            edges = []
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                row ^= bit
                edge = len(variables)
                variables.append(c)
                edges.append(edge)
                var_edges[c].append(edge)
                matrix[r, c] = 1
            row_edges.append(edges)

        ne = len(variables)
        variables = np.asarray(variables, dtype=np.intp)
        rd, vd = max(map(len, row_edges)), max(map(len, var_edges))
        rmap = np.full((len(rows), rd), ne, dtype=np.intp)
        vmap = np.full((width, vd), ne, dtype=np.intp)
        for r, edges in enumerate(row_edges):
            rmap[r, :len(edges)] = edges
        for c, edges in enumerate(var_edges):
            vmap[c, :len(edges)] = edges
        valid = rmap != ne

        target = (matrix @ errors) & 1
        target_sign = np.where(target != 0, -1.0, 1.0)
        prior = math.log(0.975/0.025)
        incoming = np.full((ne+1, shots), prior, dtype=float)
        incoming[ne] = 30.0
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        last = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        slots = np.arange(rd)[None, :, None]

        for iteration in range(40):
            gathered = incoming[rmap]
            signs = np.where(gathered >= 0, 1.0, -1.0)
            absolute = np.abs(gathered)
            smallest_at = np.argmin(absolute, axis=1)
            first = np.min(absolute, axis=1)
            second = np.partition(absolute, 1, axis=1)[:, 1, :]
            excluded = np.where(
                slots == smallest_at[:, None, :],
                second[:, None, :], first[:, None, :])
            parity = np.prod(signs, axis=1)*target_sign
            outgoing = 0.80*excluded*signs*parity[:, None, :]
            messages[rmap[valid]] = outgoing[valid]
            messages[ne] = 0.0

            posterior = prior + messages[vmap].sum(axis=1)
            last = (posterior < 0).astype(np.uint8)
            if iteration % 4 == 3:
                settled = (
                    np.all(((matrix @ last) & 1) == target, axis=0)
                    & active)
                final[:, settled] = last[:, settled]
                active[settled] = False
                if not active.any():
                    break

            extrinsic = posterior[variables] - messages[:ne]
            incoming[:ne] = (
                0.20*incoming[:ne]
                + 0.80*np.clip(extrinsic, -20.0, 20.0))
            incoming[ne] = 30.0

        final[:, active] = last[:, active]
        different = final != errors
        failed = np.any(different, axis=0) | active
        return (int(failed.sum()), int(active.sum()),
                int((different & failed[None, :]).sum()))

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        for cost, trial, grid, rows in pool:
            bad, boundary = small_kernel_and_boundary(rows, widths[side])
            d = 6 if bad else kernel_upper_bound(rows, widths[side])
            probe = decoder_probe(rows, side)
            entries.append((cost, d, bad, boundary, probe, trial, grid))
        finalists.append(entries)

    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, ad, abad, abound, ap, at, ag = a
            bc, bd, bbad, bbound, bp, bt, bg = b
            da, db = min(ad, 12), min(bd, 12)
            # Keep the protection floor, then compare decoder ambiguity.
            # Five A copies and two B copies motivate these opportunity
            # weights; they are NOT independent circuit-failure rates.
            af, bf = B_SHAPE[1]*ap[0], A_SHAPE[1]*bp[0]
            au, bu = B_SHAPE[1]*ap[1], A_SHAPE[1]*bp[1]
            key = (
                int(abad or bbad), int(min(da, db) < 10),
                max(af, bf), af + bf,
                max(au, bu), au + bu,
                ap[2]/widths[0] + bp[2]/widths[1],
                ac[0] + bc[0],
                max(abound, bbound), abound + bbound,
                ac[1] + bc[1], -min(da, db),
                abs(da-db), ac[2] + bc[2], at, bt,
            )
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    fallback = None
    examined = 0
    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or k != 144:
            continue
        if not css_ok(hx, hz):
            continue
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        # Confirm that cached actions screen exactly the components in
        # the submitted full matrices, including both lift conventions.
        ar = [
            sum(((hx[t] >> (j * B_SHAPE[1] * q)) & mask) << (j * q)
                for j in range(A_SHAPE[1]))
            for t in range(q)
        ]
        br = [row & ((1 << widths[1]) - 1)
              for row in hz[:B_SHAPE[0] * q]]
        assert ar == component_rows(a, 0)
        assert br == component_rows(b, 1)
        assert gf2_rank(ar) == q
        candidate = {"idx": idx, "design": design, "mode": mode,
                     "cells": {"A": a, "B": b}, "k": k,
                     "screen": key, "small_logical_safe": False,
                     "contract": (aw, bp, b_palette)}
        if fallback is None:
            fallback = candidate
        if not has_small_logical(hx, hz, n):
            candidate["small_logical_safe"] = True
            return candidate
        examined += 1
        if examined >= 12:
            break

    if fallback is not None:
        return fallback
    raise ValueError("no binary octahedral pair has verified k=144 in cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s=12 implies equal circuit reliability: parent slot16 scored 6.389 with 42 Z failures and 145 Z LSD resolutions, while slot21 scored 6.931 with 8 Z failures and 28 Z LSD resolutions.
# REFUTED: Equal (n,k,w,d2s) fixes circuit reliability: measured dihedral slots2/4 both had (560,112,9,12), but scored 7.270/6.931 with two/eight failed X shots out of 200000.
# REFUTED: A unit-trinomial architecture by itself eliminates basis imbalance: GL(2,3) slot6 had k144,d2s12,depth14 but failed 34 X versus 3 Z shots per 200000, with 205 versus 6 stage-three resolutions.
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
