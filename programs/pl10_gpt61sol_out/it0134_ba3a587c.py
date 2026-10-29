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
CONCEPT = ("Nonabelian C5 semidirect C4 with inversion action, order 20, shapes (2,5)x(2,5): balanced degree-three/four columns, exclusion of relative involutions and lower extraction exposure should beat the approximately 5.8 frontier fitness in cell 0.300.")
GROUP = ("Jump from cyclic C17 to C5 semidirect C4, isomorphic to Dic5. The parity action C4->Aut(C5) is a homomorphism, giving an associative nonabelian group of order 20. RUN STATE records approximately 5.6 for the centre-two/eight-class order-20 bed and 5.8 for cyclic C20 on these shapes; these motivate refined labels, not a demonstrated nonabelian advantage.")
SHAPES = ("Equal (2,5) components have 100 variables and 40 checks each. n=580 and kb=180 give guaranteed rate 9/29=0.310345 in cell 0.300. Every count profile has augmentation rank two; verified component ranks require full rank, and complete CSS ranks retain 180<=k<=188. Parent failures implicated several Acol/Bcol and Arow/Brow blocks, including Bcol4 and Arow1/Brow2; their small sample sizes do not establish one causal hotspot, so both components receive distributed protection.")
FILLING = ("Twenty explicit support pairs prescribe column weights three/four, row totals seven/eight and cells of at most three terms. HX and HZ row weights are at most twelve; six-round CNOTs are 25200..26880 versus the parent's measured 28560..29376. Fresh labels exclude relative orders one/two, with selected doubletons testing orders five, four or ten. Isolated trinomials must be invertible under the verified lift. Independently anneal component cycles, test kernels through weight six, and screen sampled witness multiplicity and boundary concentration. Parent slot 53 had eight X failures/200000 at d2s=14 and depth 14, versus two for slots 36/38 at the same d2s and depth 13; reduced exposure and ambiguity are hypotheses about circuit pL, not certificates.")
FAMILY_NAME = "r030_dic5_25x25_orderpalette_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect C4 requires order 20")
    # y*x*y^-1=x^-1; x^5=y^4=1.
    # Parity defines a homomorphism C4 -> Aut(C5).
    m, r = 5, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        sign = -1 if g[1] % 2 else 1
        return ((-sign * g[0]) % m, (-g[1]) % r)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Design supports, optimize fresh labels, then verify the full code."""
    import math

    q = G.q
    mode = (idx // 20) % 3
    assert q == 20
    assert A_SHAPE == B_SHAPE == (2, 5)

    # Columns specify both row counts. Every row totals seven/eight;
    # every column totals three/four, so BOTH CSS budgets are <=12.
    column_profiles = (
        ((2,1),(2,1),(1,2),(1,2),(1,2)),
        ((2,1),(2,1),(1,2),(1,2),(2,2)),
        ((2,1),(1,2),(1,2),(1,2),(3,1)),
        ((3,0),(2,1),(1,2),(1,2),(1,2)),
        ((3,0),(1,2),(1,2),(1,2),(1,2)),
        ((3,0),(0,3),(2,1),(1,2),(1,2)),
        ((3,0),(0,3),(2,1),(1,2),(2,2)),
        ((3,0),(1,2),(1,2),(1,2),(2,2)),
    )
    profiles = [
        [[column[i] for column in profile] for i in range(2)]
        for profile in column_profiles
    ]
    # Unordered support pairs avoid constructing A/B exchange copies.
    briefs = (
        (0,0),(0,1),(0,2),(1,1),(1,2),(2,2),
        (0,3),(0,4),(1,3),(1,4),(2,3),(2,4),
        (3,3),(3,4),(4,4),(0,5),(1,5),(2,5),
        (0,6),(0,7),
    )
    pa, pb = briefs[idx % len(briefs)]
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]

    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= min(3, CELL_MAX)
               for grid in (sa, sb) for row in grid for w in row)
    for support in (sa, sb):
        augmentation = [
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support
        ]
        assert gf2_rank(augmentation) == 2

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Use the verified construction with an identity companion.
        # The first sector is precisely the required component matrix.
        identity = [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, identity)
        else:
            _, rows, _ = lifted_matrices(G, identity, grid)
        nc = len(grid[0]) * q
        mask = (1 << nc) - 1
        return [row & mask for row in rows], nc

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def cycle_energy(rows, nc):
        nr = len(rows)
        adj = [0] * nr
        four = 0
        for i in range(nr):
            for j in range(i):
                common = (rows[i] & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adj[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adj[i] & adj[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = 0
        for col in columns(rows, nc):
            d = col.bit_count()
            stars += d * (d - 1) * (d - 2) // 6
        # Exact six-cycle count without four-cycles; otherwise a heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 4.0) / q, four, six

    def boundary_scan(cols):
        """Exact existence test for kernel words of weights 1..6.

        Collisions between subsets of size at most three cover all six
        weights. Boundary statistics are only trapping-set heuristics.
        """
        nc = len(cols)
        seen = {0: 0}
        bad = False
        risk = 0
        loads = [0] * (nc // q)

        def record(syndrome, word):
            nonlocal bad
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif old != word:
                bad = True

        for i, col in enumerate(cols):
            record(col, 1 << i)
        for i in range(nc):
            for j in range(i):
                record(cols[i] ^ cols[j], (1 << i) | (1 << j))
        for i in range(nc - 2):
            for j in range(i + 1, nc - 1):
                pair = cols[i] ^ cols[j]
                word = (1 << i) | (1 << j)
                for t in range(j + 1, nc):
                    syndrome = pair ^ cols[t]
                    record(syndrome, word | (1 << t))
                    weight = syndrome.bit_count()
                    if weight <= 6:
                        cost = (8 - weight) ** 2
                        risk += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[t // q] += cost
        return int(bad), risk / nc, max(loads) / q

    def kernel_exposure(cols, trials):
        """Verified sampled kernel words, not a distance certificate.

        Random fundamental circuits and their pair XORs expose short
        words missed by a single systematic basis.
        """
        nc = len(cols)
        order = list(range(nc))
        distance = nc + 1
        small = set()
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome = cols[c]
                word = 1 << c
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
                    w = word.bit_count()
                    distance = min(distance, w)
                    if w <= 12:
                        small.add(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    v = word ^ other
                    if not v:
                        continue
                    w = v.bit_count()
                    distance = min(distance, w)
                    if w <= 12:
                        small.add(v)
            if distance <= 6:
                break
        return distance, len(small)

    # Compute orders from the declared group law, not its encoding.
    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = order
                break
        assert G.idx(g) in orders

    allowed = {}
    for g in G.elements:
        gi = G.inv(g)
        allowed[G.idx(g)] = {
            G.idx(h) for h in G.elements
            if orders[G.idx(G.mul(gi, h))] >= 4
        }
    inverse_blocks = {}

    def draw_cell(w, side, i, j, supports):
        """Fresh order-constrained cells; no saved-label transformations."""
        if w <= 1:
            return rng.sample(G.elements, w)
        isolated = (w == 3 and
                    sum(supports[r][j] > 0
                        for r in range(len(supports))) == 1)
        marker = (idx % 20 + j + 2 * i + int(side == "B")) % 4

        for attempt in range(128):
            if w == 2:
                anchor = G.pick(rng)
                relatives = [
                    g for g in G.elements if orders[G.idx(g)] >= 4
                ]
                if marker == 0:
                    # Extra harness batches change prescribed relative
                    # orders, not a relabeling of a stored component.
                    desired = (5, 4, 10)[mode]
                    relatives = [
                        g for g in relatives
                        if orders[G.idx(g)] == desired
                    ]
                relative = rng.choice(relatives)
                cell = [anchor, G.mul(anchor, relative)]
            else:
                cell = rng.sample(G.elements, w)

            if any(G.idx(cell[b]) not in allowed[G.idx(cell[a])]
                   for a in range(w) for b in range(a)):
                continue
            if isolated:
                gi = G.inv(cell[0])
                normalized = [G.mul(gi, g) for g in cell]
                signature = tuple(sorted(G.idx(g) for g in normalized))
                if signature not in inverse_blocks:
                    probe = lifted_matrices(
                        G, [[normalized]], [[[]]])[0]
                    inverse_blocks[signature] = gf2_rank(probe) == q
                if not inverse_blocks[signature]:
                    continue
            return cell
        raise ValueError("No invertible order-legal cell in this palette")

    def design_component(supports, side):
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        locals_found = []
        for restart in range(6):
            grid = [[draw_cell(w, side, i, j, supports)
                     for j, w in enumerate(row)]
                    for i, row in enumerate(supports)]
            rows, nc = component(grid, side)
            current = cycle_energy(rows, nc)
            local = copy_grid(grid)
            local_stats = current

            for step in range(260):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(
                    supports[i][j], side, i, j, supports)
                trial_rows, _ = component(grid, side)
                proposed = cycle_energy(trial_rows, nc)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.98 ** step) + 0.12
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    current = proposed
                    if (current[0] < local_stats[0] or
                            (current[0] == local_stats[0]
                             and rng.random() < 0.06)):
                        local = copy_grid(grid)
                        local_stats = current
                else:
                    grid[i][j] = old

            rows, nc = component(local, side)
            # Full augmentation rank permits, but does not establish,
            # full component rank over this characteristic-two group ring.
            if gf2_rank(rows) != len(rows):
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:3]:
            cols = columns(rows, nc)
            bad, risk, peak = boundary_scan(cols)
            distance, low = kernel_exposure(cols, 64)
            if mode == 1:
                refinement = (risk, peak, low)
            elif mode == 2:
                refinement = (peak, low, risk)
            else:
                refinement = (low, peak, risk)
            # Beyond the useful witness threshold, reduce ambiguity and
            # extraction-graph cycles rather than reward missed witnesses.
            key = (bad, -min(distance, 10), stats[1],
                   refinement, stats[2])
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "low": low, "bad": bad, "risk": risk,
                "peak": peak, "four": stats[1], "six": stats[2],
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe component filling")
        return finalists

    apool = design_component(sa, "A")
    bpool = design_component(sb, "B")
    winner = None
    winning_key = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # Only complete evaluator-convention CSS ranks determine k.
            if n != 580 or not (180 <= k <= 188):
                continue
            key = (
                max(a["bad"], b["bad"]),
                -min(a["d"], b["d"], 10),
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["low"] + b["low"],
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]),
                a["risk"] + b["risk"],
                a["six"] + b["six"],
            )
            if winning_key is None or key < winning_key:
                winning_key = key
                winner = (A, B, hx, hz, k)

    if winner is None:
        raise ValueError("No finalist retained the 0.300 rate band")
    A, B, hx, hz, k = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert 11 * 580 <= 40 * k < 13 * 580
    return {"idx": idx, "architecture": (pa, pb), "palette": mode,
            "k": k, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: d2s=14 ensures near-ceiling circuit memory on the C17 bed: parent slot 53 had eight X failures/200000 and s=6.806, while slots 36/38 at the same d2s had two worse-basis failures and s=7.145.
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
