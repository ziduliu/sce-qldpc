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
CONCEPT = ("C4 semidirect C4 with inversion action, order 16, shapes (3,5)x(3,5): balanced three-row protection and rank-screened sparse supports should exceed fitness 7.3634 by reducing collective memory failures without increasing extraction exposure.")
GROUP = ("Keep the construction rule C_m semidirect C4 with inversion. The new factor 34 makes order 36 illegal; m=4 gives order 16 and n=544. The related Z17 (3,5)x(3,5) bed reached about 7.2, motivating three-row components but not proving an advantage for this untried nonabelian bed.")
SHAPES = ("Equal (3,5) components each have 80 variables and 48 checks. The guaranteed bound is kb=64, rate 2/17=0.11765; complete verified ranks instead require 80<=k<=94, giving rates 0.14706..0.17279 in cell 0.150. Parent slot32 had two Z failures flipping 54 logicals across 44 of 82, with multiple Acol/Bcol data and Acol/Brow check blocks implicated. These sparse maps motivate distributed redundancy, not a unique-hotspot claim.")
FILLING = ("Twenty primary sparse support contracts vary augmentation rank, zero/doubleton incidence and trinomial reinforcement; extra harness slots receive unused support pairs. Search fresh labels whose within-cell relative orders are four, and explicitly enumerate alternative count contracts if the first misses verified 80<=k<=94. Every component column has degree three..five and row degree at most seven, so both CSS weight sums are <=12. Six-round CNOT count is 768 times the total protograph term count. Retain verified-lift annealing, connectivity, exact component-kernel screening through weight six, and sampled witness/boundary heuristics; use symmetric capped protection thresholds of fourteen. A rate-safe seed is retained so optimization cannot discard the only legal filling. These structural criteria neither certify ASC scheduling nor substitute for measured circuit pL.")
FAMILY_NAME = "r015_c4sc4inv_35x35_sparse_rankbalanced_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same family: C_m semidirect C4 with inversion action.
    # The new size factor 34 requires q<18; m=4 gives q=16.
    assert q == 16
    m, r = 4, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        sign = -1 if g[1] % 2 else 1
        return ((-sign * g[0]) % m, (-g[1]) % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


# Column degrees 3..5; row degrees 4..7. Every pair obeys 7+5<=12.
# Parity ranks deliberately vary: 1,2,2,2,3,1,1,1,1,2,2,3.
_SUPPORT_PROFILES = (
    ((1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1)),
    ((2,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1)),
    ((2,2,1,1,1), (1,1,1,1,1), (1,1,1,1,1)),
    ((2,1,1,1,1), (2,1,1,1,1), (1,1,1,1,1)),
    ((2,1,1,1,1), (1,2,1,1,1), (1,1,1,1,1)),
    ((3,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1)),
    ((0,1,1,1,1), (2,1,1,1,1), (2,1,1,1,1)),
    ((0,2,1,1,1), (2,0,1,1,1), (2,2,1,1,1)),
    ((0,2,2,1,1), (2,0,2,1,1), (2,2,0,1,1)),
    ((0,1,1,1,1), (2,1,1,1,1), (1,1,1,1,1)),
    ((0,1,1,1,1), (1,1,1,1,1), (3,1,1,1,1)),
    ((0,1,1,1,1), (1,2,1,1,1), (2,0,1,1,1)),
)

# Fifteen unordered parity-rank-two contracts, then five mixed-rank
# contracts. Equal components are not duplicated by an A/B exchange.
_PROFILE_PAIRS = (
    (1,1), (1,2), (1,3), (1,9), (1,10),
    (2,2), (2,3), (2,9), (2,10),
    (3,3), (3,9), (3,10),
    (9,9), (9,10), (10,10),
    (0,1), (0,2), (1,5), (1,6), (0,4),
)


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _small_kernel_word(rows, ncols):
    """Exact meet-in-the-middle test for a nonzero kernel word <=6."""
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    # Each subset of size <=3 is visited once. Equal syndromes of two
    # different subsets give a nonzero kernel word of weight <=6.
    seen = {0}
    for value in columns:
        if value in seen:
            return True
        seen.add(value)
    for i in range(ncols):
        for j in range(i):
            value = columns[i] ^ columns[j]
            if value in seen:
                return True
            seen.add(value)
    for i in range(ncols):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for h in range(j):
                value = pair ^ columns[h]
                if value in seen:
                    return True
                seen.add(value)
    return False


def NEW_CTX(rng, G, idx):
    """Sparse three-row contracts, screened on the complete verified lift."""
    import math
    from itertools import permutations

    q = G.q
    assert q == 16
    assert A_SHAPE == B_SHAPE == (3, 5)

    def parity_rank(profile):
        return gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in profile
        ])

    def support_key(profile):
        # Remove obvious count-grid row/column permutation copies.
        # This is not a complete Tanner-graph isomorphism classifier.
        return min(
            tuple(sorted(tuple(profile[i][j] for i in p)
                         for j in range(5)))
            for p in permutations(range(3))
        )

    profile_keys = [support_key(p) for p in _SUPPORT_PROFILES]
    parity = [parity_rank(p) for p in _SUPPORT_PROFILES]
    book = list(_PROFILE_PAIRS)
    used = {
        tuple(sorted((profile_keys[a], profile_keys[b])))
        for a, b in book
    }
    extras = []
    for a in range(len(_SUPPORT_PROFILES)):
        for b in range(a, len(_SUPPORT_PROFILES)):
            signature = tuple(sorted((profile_keys[a], profile_keys[b])))
            if signature in used:
                continue
            if parity[a] == parity[b] == 3:
                continue
            sa0, sb0 = _SUPPORT_PROFILES[a], _SUPPORT_PROFILES[b]
            if _grid_weights(sa0, sb0) > W_MAX:
                continue
            used.add(signature)
            extras.append((a, b))
    extras.sort(key=lambda pair: (
        abs(parity[pair[0]] + parity[pair[1]] - 4),
        sum(sum(row) for p in pair for row in _SUPPORT_PROFILES[p]),
        pair
    ))
    book.extend(extras)
    if idx >= len(book):
        raise ValueError("No unused sparse support contract for this slot")

    pa, pb = book[idx]
    sa = [list(row) for row in _SUPPORT_PROFILES[pa]]
    sb = [list(row) for row in _SUPPORT_PROFILES[pb]]
    counts = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX
               for grid in counts.values() for row in grid for w in row)

    # Preserve the family, not the parent's order palette: this group
    # has exponent four. Within-cell differences of order two are banned.
    # Supports and labels are designed afresh, never transformed saves.

    orders = []
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                orders.append(order)
                break
        else:
            raise ValueError("Element order exceeded the group order")
    relative = [
        [orders[G.idx(G.mul(G.inv(g), h))] for h in G.elements]
        for g in G.elements
    ]

    def allowed(side, i, j):
        return (4,)

    def draw_cell(weight, palette):
        if not weight:
            return []
        for _ in range(24):
            chosen = [rng.randrange(q)]
            for t in range(1, weight):
                available = [
                    h for h in range(q)
                    if all(relative[g][h] in palette for g in chosen)
                ]
                if not available:
                    break
                chosen.append(rng.choice(available))
            if len(chosen) == weight:
                return [G.elements[t] for t in chosen]
        raise ValueError("Unable to realize the prescribed relative orders")

    def draw_grid(side):
        return [
            [draw_cell(w, allowed(side, i, j))
             for j, w in enumerate(row)]
            for i, row in enumerate(counts[side])
        ]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        # Zero probes preserve precisely the scaffold's left/right actions.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= marker
                row ^= bit
        return result

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            first = (row & -row).bit_length() - 1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(t) for t in range(width)}) == 1

    def cycle_energy(rows, width):
        four = 0
        adjacency = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            lower = adjacency[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in columns(rows, width)))
        # Exact six-cycles if C4=0; otherwise only a proposal heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 8.0) / q, four, six

    # Find and retain a complete rate-safe seed before component
    # optimization. Nonabelian component deficits are not substituted
    # for the ranks of the submitted HX/HZ.
    rate_seed = None
    for offset in range(len(book)):
        pa, pb = book[(idx + offset) % len(book)]
        sa = [list(row) for row in _SUPPORT_PROFILES[pa]]
        sb = [list(row) for row in _SUPPORT_PROFILES[pb]]
        counts = {"A": sa, "B": sb}
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        for trial in range(32):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 544 or not 80 <= k <= 94:
                continue
            ar, aw = component("A", A)
            br, bw = component("B", B)
            ae, be = cycle_energy(ar, aw), cycle_energy(br, bw)
            key = (max(ae[1], be[1]), ae[1] + be[1],
                   ae[2] + be[2], -k)
            if rate_seed is None or key < rate_seed[0]:
                rate_seed = (key, clone(A), clone(B), hx, hz, n, k)
        if rate_seed is not None:
            break
    if rate_seed is None:
        raise ValueError("Sparse support bank missed verified cell 0.150")
    assert _grid_weights(sa, sb) <= W_MAX
    selected_contract = (pa, pb)

    def boundary_exposure(cols):
        """Exact small-boundary triple statistic, not a trapping certificate."""
        total = 0
        loads = [0] * (len(cols) // q)
        for i in range(len(cols)):
            for j in range(i):
                pair = cols[i] ^ cols[j]
                for h in range(j):
                    weight = (pair ^ cols[h]).bit_count()
                    if weight <= 4:
                        cost = (6 - weight) ** 2
                        total += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[h // q] += cost
        return total / len(cols), max(loads) / q

    def sampled_words(cols, side):
        """Verified fundamental/pair-XOR words; minimum is an upper bound.

        Use the same permutation stream for finalists in this slot.
        Multiplicity and block exposure refer only to sampled witnesses.
        """
        srng = random.Random(73129 + 1009 * idx + (side == "B"))
        width = len(cols)
        permutation = list(range(width))
        minimum = width + 1
        short = set()
        threshold = 14
        for trial in range(40):
            srng.shuffle(permutation)
            pivots = {}
            basis = []
            for c in permutation:
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

            for i, word in enumerate(basis):
                candidates = [word]
                candidates.extend(word ^ other for other in basis[:i])
                for v in candidates:
                    if not v:
                        continue
                    weight = v.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= threshold and len(short) < 8192:
                        short.add(v)
            if minimum <= 6:
                break

        mask = (1 << q) - 1
        loads = [0] * (width // q)
        very_short = 0
        for word in short:
            very_short += int(word.bit_count() <= 12)
            for j in range(len(loads)):
                loads[j] += int(bool((word >> (j * q)) & mask))
        return minimum, very_short, len(short), max(loads)

    def portfolio(side):
        supports = counts[side]
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        pool = []
        seen = set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            coarse = (stats[1], stats[2])
            if len(pool) >= 6 and coarse > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((coarse, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        seed_grid = rate_seed[1] if side == "A" else rate_seed[2]
        seed_rows, seed_width = component(side, seed_grid)
        remember(seed_grid, seed_rows, cycle_energy(seed_rows, seed_width))

        for restart in range(3):
            grid = draw_grid(side)
            rows, width = component(side, grid)
            current = cycle_energy(rows, width)
            remember(grid, rows, current)
            for step in range(260):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(supports[i][j], allowed(side, i, j))
                trial_rows, _ = component(side, grid)
                proposed = cycle_energy(trial_rows, width)
                delta = proposed[0] - current[0]
                temperature = 10.0 * (0.985 ** step) + 0.08
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    rows, current = trial_rows, proposed
                    remember(grid, rows, current)
                else:
                    grid[i][j] = old

        finalists = []
        for coarse, serial, grid, rows in pool:
            width = len(grid[0]) * q
            cols = columns(rows, width)
            bad = int(_small_kernel_word(rows, width))
            disconnected = int(not connected(rows, width))
            d, low12, low, load = sampled_words(cols, side)
            risk, peak = boundary_exposure(cols)
            four = coarse[0]
            six = _six_cycles(rows)
            target = 14
            key = (disconnected, bad, max(0, target - d),
                   low12, low, load, peak, risk, four, six)
            finalists.append({
                "grid": grid, "key": key, "bad": bad,
                "disconnected": disconnected, "d": d,
                "low12": low12, "low": low, "load": load,
                "risk": risk, "peak": peak,
                "four": four, "six": six,
                "deficit": len(rows) - gf2_rank(rows),
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists

    apool, bpool = portfolio("A"), portfolio("B")
    winner = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 544 or not 80 <= k <= 94:
                continue
            # Equal-length, three-row components receive equal pressure.
            # Sampled protection saturates at fourteen; it is not fitness.
            key = (
                a["disconnected"] + b["disconnected"],
                a["bad"] + b["bad"],
                max(0, 14 - a["d"]) + max(0, 14 - b["d"]),
                max(a["low12"], b["low12"]),
                a["low12"] + b["low12"],
                a["low"] + b["low"],
                max(a["load"], b["load"]),
                max(a["peak"], b["peak"]),
                a["risk"] + b["risk"],
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["six"] + b["six"],
                -k,
            )
            if winner is None or key < winner[0]:
                winner = (key, A, B, hx, hz, n, k)

    # Independent component optimization may discard compatible ranks.
    # The already verified complete seed prevents that from voiding a slot.
    if winner is None:
        winner = rate_seed

    key, A, B, hx, hz, n, k = winner
    assert n == 544 and 80 <= k <= 94
    assert 8 * k > n and 40 * k < 7 * n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    expected_cnots = (
        (rb + cb) * sum(map(sum, sa)) +
        (ra + ca) * sum(map(sum, sb))
    ) * q * 6
    assert sum(row.bit_count() for row in hx + hz) * 6 == expected_cnots
    return {"A": A, "B": B, "design": (idx, selected_contract),
            "k": k, "screen": key, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the already screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Excluding component kernel words <=6 and minimizing component four/six cycles guarantees ASC measurability: current slot 14 passed these screens but its X schedule timed out after 120 seconds.
# REFUTED: Equal d2s predicts equal circuit reliability: current slots 1 and 11 both had d2s=15, but their worse bases had zero and 13 failures in 200000 shots.
# REFUTED: d2s=16 ensures near-ceiling memory in this order-36 bed: parent slot 11 had 45 Z failures/200000 and s=6.186, whereas slot 58 had one failure per basis and s=7.225.
# REFUTED: Matching k, d2s, check weight, depth and CNOT count guarantees identical circuit reliability: order-36 slots12 and32 both had k=82, d2s=15, w=11, depth13 and 30672 CNOTs, but slot12 had zero failures while slot32 had two Z failures and 54 collective logical flips.
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
