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
CONCEPT = ("C17 semidirect C4, action 4, order 68, A=(1,2), B=(1,3): adjacent-quotient cubic cells with packed pair differences should produce five measured distance readings of at least twelve in cell 0.300.")
GROUP = ("Jump from GL(2,3) to the genuinely different faithful semidirect product C17:C4. The action 4 has order four modulo 17, and order 68 gives n=476. No order-68 bed appears in BEDS. The C37:C3 bed's strong measured readings motivate testing another faithful action, not a claim that those readings transfer.")
SHAPES = ("Target 0.300 rather than the parent's 0.250 cell: n=7*68=476, kb=2*68=136 and guaranteed rate 2/7. A has 136 variables/68 checks; B has 204 variables/68 checks. All five measured parent codes retained weight-twelve X and Z minima entirely in sector 1, using both A columns and multiple B columns. Replace both component geometries; verified ranks retain [0.275,0.325).")
FILLING = ("Twenty rotation-gap-ratio cards fill every cell with three distinct terms, giving HX weight nine and HZ weight twelve. Adjacent C4 cosets exclude relative involutions and give noncommuting relatives. Pack directed pair differences, then verify component four-cycle exclusion. Select by six-cycle structure, exact short-kernel tests and progressively deeper component/full-logical witness probes using only verified lifts. Full Tanner walk fingerprints separate retained designs. Search readings remain upper bounds, not certificates.")
FAMILY_NAME = "r030_C17C4_action4_12x13_cubic_gapratio_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 68

def custom_ops(q):
    # x^17=y^4=1, y*x*y^-1=x^4.
    # 4^2=-1 and 4^4=1 modulo 17.
    if not valid_q(q):
        raise ValueError("C17:C4 requires order 68")
    action = (1, 4, 16, 13)
    els = [(a, b) for b in range(4) for a in range(17)]

    def red(g):
        return (g[0] % 17, g[1] % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 17, (b+d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((-action[bi]*a) % 17, bi)

    def idx(g):
        a, b = red(g)
        return 17*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)

from functools import lru_cache

# Twenty prescribed ratio architectures. Absolute scales and offsets
# are optimized internally, not used to manufacture orbit copies.
_A_GAP_CARDS = ((1,1), (1,2), (1,3), (1,5), (1,6))
_B_GAP_CARDS = ((1,2,3), (1,2,5), (1,3,6), (1,5,7))


def _packed_grid(rng, G, gaps, side):
    """Pack pair orbits appropriate to each component action.

    The filter is group-theoretic. Four-cycle exclusion is separately
    checked on slices of the scaffold's verified matrices.
    """
    scale = rng.randrange(1, 17)
    for restart in range(24):
        used = set()
        row = [None] * len(gaps)
        order = list(range(len(gaps)))
        rng.shuffle(order)
        for j in order:
            gap = scale*gaps[j] % 17
            for attempt in range(128):
                anchor = G.pick(rng)
                sign = rng.choice((1, 3))
                d = gap if rng.randrange(2) else (-gap) % 17
                cell = [
                    anchor,
                    G.mul(anchor, (d, 0)),
                    G.mul(anchor, (rng.randrange(17), sign)),
                ]
                if side == "A":
                    differences = {
                        G.mul(g, G.inv(h))
                        for g in cell for h in cell if g != h
                    }
                else:
                    differences = {
                        G.mul(G.inv(g), h)
                        for g in cell for h in cell if g != h
                    }
                if len(differences) != 6 or differences & used:
                    continue
                used.update(differences)
                row[j] = cell
                break
            if row[j] is None:
                break
        if all(cell is not None for cell in row):
            return [row]
    return None


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials, stabilizers=()):
    """Actual kernel witnesses; with stabilizers, actual logical witnesses.

    Recombination and descent preserve the supplied kernel. Membership
    in the stabilizer row space is tested exactly before recording a
    full-code witness. A sentinel is not a distance certificate.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(word):
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    best = len(columns)+1
    memory = []
    order = list(range(len(columns)))

    def observe(word):
        nonlocal best
        if word and word.bit_count() < best:
            if not basis or outside(word):
                best = word.bit_count()

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
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
                observe(word)

        short = sorted(set(words+memory), key=int.bit_count)[:48]
        discovered = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                observe(combined)
                if combined and combined.bit_count() <= best+4:
                    discovered.add(combined)

        reducers = words+memory+list(stabilizers)
        for seed in sorted(discovered, key=int.bit_count)[:8]:
            word = seed
            for step in range(4):
                chosen, weight = word, word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    if combined and combined.bit_count() < weight:
                        chosen, weight = combined, combined.bit_count()
                if chosen == word:
                    break
                word = chosen
                observe(word)
                discovered.add(word)
        memory = sorted(discovered, key=int.bit_count)[:48]
        if best <= 4:
            break
    return best


def _no_kernel_le6(columns):
    """Exact component test, not a quantum-distance certificate.

    Every support of weight <=6 partitions into two different subsets
    of size <=3. A syndrome collision therefore detects such a support.
    Only existence is tested; no witness census is claimed.
    """
    seen = {0}
    for syndrome in columns:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _component_data(hx, hz, q):
    """Restrict the VERIFIED lift; do not implement regular actions."""
    mask = (1 << q) - 1
    cb = B_SHAPE[1]
    ar = [
        (row & mask) | (((row >> (cb*q)) & mask) << q)
        for row in hx[:q]
    ]
    bm = (1 << (cb*q)) - 1
    br = [row & bm for row in hz[:B_SHAPE[0]*q]]
    return ar, br, _columns(ar, 2*q), _columns(br, cb*q)


def _six_cycles(columns):
    """Exact six-cycle count for a four-cycle-free cubic component."""
    height = max(s.bit_length() for s in columns)
    adjacency = [0] * height
    for syndrome in columns:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length()-1)
            syndrome ^= bit
        assert len(neighbors) == 3
        for a in neighbors:
            for b in neighbors:
                if a != b:
                    adjacency[a] |= 1 << b
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        higher = neighbors >> (a+1)
        while higher:
            bit = higher & -higher
            b = a+1+bit.bit_length()-1
            triangles += ((neighbors & adjacency[b]) >> (b+1)).bit_count()
            higher ^= bit
    # Every cubic variable contributes one same-variable triangle.
    return triangles-len(columns)


def _walk_signature(rows, width):
    """Full Tanner invariants under arbitrary check/qubit permutations.

    Different signatures prove non-isomorphism; equal signatures can
    falsely reject distinct codes and are conservatively discarded.
    """
    columns = _columns(rows, width)
    adjacency = [{} for _ in rows]
    for syndrome in columns:
        active = []
        while syndrome:
            bit = syndrome & -syndrome
            active.append(bit.bit_length()-1)
            syndrome ^= bit
        for t, a in enumerate(active):
            for b in active[:t]:
                adjacency[a][b] = adjacency[a].get(b, 0)+1
                adjacency[b][a] = adjacency[b].get(a, 0)+1

    overlaps = {}
    edges = 0
    triangle_walks = fourth_walks = 0
    for a, neighbors in enumerate(adjacency):
        for b, value in neighbors.items():
            if b < a:
                overlaps[value] = overlaps.get(value, 0)+1
                edges += 1
        walks = {}
        for b, ab in neighbors.items():
            for c, bc in adjacency[b].items():
                walks[c] = walks.get(c, 0)+ab*bc
        triangle_walks += sum(
            value*walks.get(b, 0) for b, value in neighbors.items())
        fourth_walks += sum(value*value for value in walks.values())
    overlaps[0] = len(rows)*(len(rows)-1)//2-edges
    return (
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(col.bit_count() for col in columns)),
        tuple(sorted(overlaps.items())),
        triangle_walks, fourth_walks,
    )


def _code_signature(item):
    return (item["k"], tuple(sorted((
        _walk_signature(item["hx"], item["n"]),
        _walk_signature(item["hz"], item["n"]),
    ))))


def _candidate_merit(item):
    # Component readings are selection heuristics, not certified
    # quantum distances. Full readings count only non-stabilizers.
    d = min(item["da"], item["db"], item.get("full", item["n"]+1))
    return (
        item["k"]*d*d/item["n"], d,
        item.get("clean6", 0), item["da"]+item["db"],
        -sum(item["six"]), -max(item["six"]),
    )


@lru_cache(maxsize=1)
def _design_bank(G):
    """Pure deterministic optimization of invariant-distinct card designs."""
    assert G.q == 68
    sa, sb = [[3,3]], [[3,3,3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    bank, seen = [], set()

    for idx in range(N_SLOTS):
        card = idx % 20
        ag = _A_GAP_CARDS[card // 4]
        bg = _B_GAP_CARDS[card % 4]
        chosen = None

        for retry in range(6):
            local = random.Random(304170041+1009*idx+7919*retry)
            raw = []
            quota = 48+16*retry+4*(idx // 20)
            for attempt in range(quota):
                A = _packed_grid(local, G, ag, "A")
                B = _packed_grid(local, G, bg, "B")
                if A is None or B is None:
                    continue
                hx, hz, n = lifted_matrices(G, A, B)
                ar, br, ac, bc = _component_data(hx, hz, G.q)
                if _component_cycles(ar) or _component_cycles(br):
                    continue
                six = (_six_cycles(ac), _six_cycles(bc))
                structure = (sum(six), max(six), six[1], attempt)
                raw.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, ac=ac, bc=bc,
                    six=six, structure=structure))
                raw.sort(key=lambda item: item["structure"])
                del raw[12:]

            pool = []
            for item in raw:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (275*n <= 1000*k < 325*n):
                    continue
                if min(_short_floor(item["ac"]),
                       _short_floor(item["bc"])) < 5:
                    continue
                item["k"] = k
                item["da"] = _kernel_reading(item["ac"], local, 8)
                item["db"] = _kernel_reading(item["bc"], local, 10)
                # Require found component witnesses, not large sentinels.
                if (item["da"] > len(item["ac"])
                        or item["db"] > len(item["bc"])):
                    continue
                pool.append(item)

            finalists = sorted(pool, key=_candidate_merit,
                               reverse=True)[:4]
            for item in finalists:
                item["da"] = min(item["da"], _kernel_reading(
                    item["ac"], local, 32))
                item["db"] = min(item["db"], _kernel_reading(
                    item["bc"], local, 40))
                dx = _kernel_reading(
                    _columns(item["hz"], item["n"]), local, 12,
                    stabilizers=item["hx"])
                dz = _kernel_reading(
                    _columns(item["hx"], item["n"]), local, 12,
                    stabilizers=item["hz"])
                item["full"] = min(dx, dz)

            leaders = sorted(finalists, key=_candidate_merit,
                             reverse=True)[:2]
            for item in leaders:
                ca = item["da"] > 6 and _no_kernel_le6(item["ac"])
                cb = item["db"] > 6 and _no_kernel_le6(item["bc"])
                item["clean6"] = int(ca)+int(cb)
                item["da"] = min(item["da"], _kernel_reading(
                    item["ac"], local, 48))
                item["db"] = min(item["db"], _kernel_reading(
                    item["bc"], local, 64))
                dx = _kernel_reading(
                    _columns(item["hz"], item["n"]), local, 24,
                    stabilizers=item["hx"])
                dz = _kernel_reading(
                    _columns(item["hx"], item["n"]), local, 24,
                    stabilizers=item["hz"])
                item["full"] = min(item["full"], dx, dz)

            for item in sorted(finalists, key=_candidate_merit,
                               reverse=True):
                signature = _code_signature(item)
                if signature in seen:
                    continue
                assert css_ok(item["hx"], item["hz"])
                assert max(row.bit_count()
                           for row in item["hx"]+item["hz"]) <= W_MAX
                assert all(
                    len(cell) == 3 and len(set(cell)) == 3
                    for grid in (item["A"], item["B"])
                    for row in grid for cell in row)
                seen.add(signature)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row)
                          for row in item["A"]),
                    tuple(tuple(tuple(cell) for cell in row)
                          for row in item["B"]),
                    item["k"], _candidate_merit(item),
                )
                break
            if chosen is not None:
                break

        if chosen is None:
            raise ValueError("no fresh target-rate gap-ratio design")
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized design, never an orbit copy."""
    A, B, k, score = _design_bank(G)[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": score}


def CELL(rng, G, side, i, j, ctx):
    """All labels and counts were selected jointly in NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting GL(2,3) order palettes and component-through-six screens eliminate sector-1 weight-twelve logicals; all five measured parent codes retained both X and Z minima at twelve, entirely in sector 1.
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
