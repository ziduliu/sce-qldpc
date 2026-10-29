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
CONCEPT = ("For the Frobenius group (C5 x C5):C3, order 75, shapes (1,2)x(1,3), noncommuting trinomials with disjoint component check-pair orbits should beat the 0.300 frontier fitness 30.7648.")
GROUP = ("Jump from C37:C3 to an elementary-abelian normal subgroup with action T(a,b)=(-b,a-b). T^3=I and det(T-I)=3 mod 5, so the action is fixed-point-free and the group has odd order 75. This bed is absent from RUN STATE. The strong C37:C3 readings motivate another fixed-point-free C3 action without asserting distance transfer.")
SHAPES = ("Target 0.300: A=(1,2), B=(1,3) gives n=525, kb=150 and guaranteed rate 2/7. Component variable/check counts are 150/75 and 225/75; LP sectors have 450 and 75 qubits. Parent measured weight-21 minima used sector-two Arow/Brow blocks, especially Arow1 across both B rows. Replace that geometry with three B columns and require verified 0.275<=k/n<0.325.")
FILLING = ("Twenty quotient-composition cards use three distinct terms in all five cells: HX weight nine and HZ weight twelve. Require noncommuting relative labels and pack disjoint directed check-pair orbits; verify zero component four-cycles on scaffold lifts. Equal-budget kernel searches refine independently optimized components, then actual ranks and stabilizer-tested component/full-code witnesses select combinations. Combined-Tanner walk fingerprints reject emitted orbit copies. All search readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r030_affineF25c3_12x13_packed_cubic_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 75

def custom_ops(q):
    # F5^2 semidirect C3, with T=[[0,-1],[1,-1]].
    # T^2(a,b)=(b-a,-a), T^3=I; T has no nonzero fixed vector.
    if not valid_q(q):
        raise ValueError("(C5 x C5):C3 requires order 75")
    els = [(a,b,t) for t in range(3)
           for b in range(5) for a in range(5)]

    def red(g):
        return (g[0] % 5, g[1] % 5, g[2] % 3)

    def action(t, a, b):
        t %= 3
        if t == 0:
            return (a % 5, b % 5)
        if t == 1:
            return ((-b) % 5, (a-b) % 5)
        return ((b-a) % 5, (-a) % 5)

    def idx(g):
        a,b,t = red(g)
        return 25*t + 5*b + a

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        c,d = action(t, c, d)
        return ((a+c) % 5, (b+d) % 5, (t+u) % 3)

    def inv(g):
        a,b,t = red(g)
        u = (-t) % 3
        c,d = action(u, -a, -b)
        return (c,d,u)

    return dict(elements=els, ident=(0,0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _fp_basis(rows):
    pivots = {}
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    return pivots


def _fp_reduce(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return v
        v ^= pivots[p]
    return 0


def _fp_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _fp_cycles(columns):
    total = 0
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            overlap = (a & b).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _fp_witness(rng, columns, embed, stabilizers, trials=8,
                best=None, short=True, descent_rows=()):
    """Actual commuting witnesses, with optional stabilizer descent.
    A supplied best must already come from a nontrivial witness.
    The width+1 sentinel means no witness, never a distance reading."""
    width = len(columns)
    if best is None:
        best = width + 1

    def observe(word):
        nonlocal best
        if not word:
            return
        if not descent_rows and word.bit_count() >= best:
            return
        full = embed(word)
        if descent_rows and not _fp_reduce(full, stabilizers):
            # A stabilizer cannot become logical under stabilizer descent.
            return
        if descent_rows:
            # Adding checks preserves commutation and logical class.
            weight = full.bit_count()
            for _ in range(2):
                changed = False
                for row in descent_rows:
                    trial = full ^ row
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        full, weight = trial, tw
                        changed = True
                if not changed:
                    break
        weight = full.bit_count()
        if weight < best and _fp_reduce(full, stabilizers):
            best = weight

    if short:
        # Representatives of supports of size <=2 cover weights <=4.
        # Stabilizer-equivalent collisions preserve the quotient class.
        seen = {0: 0}
        for i, syndrome in enumerate(columns):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            else:
                observe(word ^ old)
        for i in range(width):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                else:
                    observe(word ^ old)
        if best <= 4:
            return best

    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)

        words.sort(key=int.bit_count)
        small = words[:24]
        for word in small:
            observe(word)
        for i, word in enumerate(small):
            for other in small[:i]:
                observe(word ^ other)
        # Short triples supplement information-set dependencies.
        for i, word in enumerate(small[:8]):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    observe(pair ^ small[h])

        for start in small[:8]:
            word = start
            for _ in range(4):
                improved, weight = word, word.bit_count()
                for other in words:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
        if best <= 4:
            break
    return best


# Five cells per card: Acol0,Acol1,Bcol0,Bcol1,Bcol2.
# Types prescribe quotient occupancies 111, 210, and 201.
# Cards are independent construction constraints, not transported labels.
# Some constraints share an orbit; final graph fingerprints reject copies.
_AFFINE_CARDS = (
    (0,0,0,0,0), (0,0,0,0,1), (0,0,0,1,1), (0,0,1,1,2),
    (0,1,0,0,0), (0,1,0,0,1), (0,1,0,1,1), (0,1,1,1,2),
    (1,1,0,0,0), (1,1,0,0,1), (1,1,0,1,1), (1,1,1,1,2),
    (1,2,0,0,0), (1,2,0,0,1), (1,2,0,1,1), (1,2,1,1,2),
    (0,2,0,0,0), (0,2,0,0,1), (0,2,0,1,1), (0,2,1,1,2),
)


def _affine_optimize_ctx(rng, G, idx):
    """Optimize packed cubic components and their actual CSS combination."""
    q = G.q
    mask = (1 << q) - 1
    card = _AFFINE_CARDS[idx % len(_AFFINE_CARDS)]
    quotient_patterns = ((0,1,2), (0,0,1), (0,0,2))
    pools = [[g for g in G.elements if g[2] == t] for t in range(3)]
    sa, sb = [[3,3]], [[3,3,3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def draw_side(side):
        models = card[:2] if side == "A" else card[2:]
        for restart in range(12):
            used = set()
            row = [None] * len(models)
            order = list(range(len(models)))
            rng.shuffle(order)
            for j in order:
                for attempt in range(96):
                    pattern = quotient_patterns[models[j]]
                    shift = rng.randrange(3)
                    cell = []
                    for t in range(3):
                        count = sum((b + shift) % 3 == t for b in pattern)
                        cell.extend(rng.sample(pools[t], count))
                    anchor = cell[0]
                    relatives = [G.mul(G.inv(anchor), g) for g in cell[1:]]
                    if G.mul(relatives[0], relatives[1]) == G.mul(
                            relatives[1], relatives[0]):
                        continue
                    differences = {
                        G._key(G.mul(g, G.inv(h)))
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

    def component_rows(hx, hz):
        # Extract both regular actions from the verified wide LP lift.
        # Fix Bcol0 for A; fix Acol0 for B.
        ar = [(row & mask) | (((row >> (3*q)) & mask) << q)
              for row in hx[:q]]
        br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
        return ar, br

    identity = lambda word: word
    skeleton_a = [[[G.e], [G.e]]]
    skeleton_b = [[[G.e], [G.e], [G.e]]]

    def component_key(item):
        return (item["d"], -item["cycles"], item["rank"])

    def optimize_side(side):
        retained = []
        width = (2 if side == "A" else 3) * q
        for _ in range(40):
            cells = draw_side(side)
            if cells is None:
                continue
            A = cells if side == "A" else skeleton_a
            B = cells if side == "B" else skeleton_b
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = component_rows(hx, hz)
            rows = ar if side == "A" else br
            columns = _fp_columns(rows, width)
            cycles = _fp_cycles(columns)
            # Packing is only a construction filter: check the real lift.
            if cycles:
                continue
            rank = gf2_rank(rows)
            reading = _fp_witness(
                rng, columns, identity, {}, trials=8)
            if reading <= 4 or reading > width:
                continue
            item = {
                "cells": cells, "columns": columns,
                "d": reading, "rank": rank, "cycles": cycles,
            }
            retained.append(item)
            retained.sort(key=component_key, reverse=True)
            del retained[4:]

        if not retained:
            raise ValueError("no packed cubic component passed kernel screening")
        # Every retained component receives the same deeper search budget.
        for item in retained:
            item["d"] = _fp_witness(
                rng, item["columns"], identity, {}, trials=48,
                best=item["d"], short=False)
        retained.sort(key=component_key, reverse=True)
        return retained[:3]

    pa, pb = optimize_side("A"), optimize_side("B")
    embed_a = lambda word: ((word & mask) | ((word >> q) << (3*q)))
    embed_b = lambda word: word

    def improve(item, trials):
        # A kernels embed as Z candidates at Bcol0.
        # B kernels embed as X candidates at Acol0.
        # Their logical classes use the actual opposite stabilizer spaces.
        item["da"] = _fp_witness(
            rng, item["ac"], embed_a, item["zs"], trials=trials,
            best=item["da"], short=False)
        item["db"] = _fp_witness(
            rng, item["bc"], embed_b, item["xs"], trials=trials,
            best=item["db"], short=False)

    def merit(item):
        d = min(item["da"], item["db"], item.get("full", item["n"]+1))
        return (item["k"] * d*d / item["n"],
                d, item["da"] + item["db"], -item["cycles"])

    combinations = []
    for a in pa:
        for b in pb:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (275*n <= 1000*k < 325*n):
                continue
            item = {
                "A": A, "B": B, "hx": hx, "hz": hz,
                "n": n, "k": k,
                "ac": a["columns"], "bc": b["columns"],
                "xs": _fp_basis(hx), "zs": _fp_basis(hz),
                "da": 2*q + 1, "db": 3*q + 1,
                "cycles": a["cycles"] + b["cycles"],
            }
            improve(item, 8)
            # Sentinels never earn a score.
            if item["da"] <= 2*q and item["db"] <= 3*q:
                combinations.append(item)

    if not combinations:
        raise ValueError("no nontrivial pair in the 0.300 rate band")
    finalists = sorted(combinations, key=merit, reverse=True)[:3]
    for item in finalists:
        improve(item, 48)
        bound = min(item["da"], item["db"])
        # Both sectors are present in these full-matrix searches.
        # Stabilizer descent can expose lighter representatives without
        # changing the logical class of the information-set word.
        dx = _fp_witness(
            rng, _fp_columns(item["hz"], item["n"]),
            identity, item["xs"], trials=6, best=bound,
            descent_rows=item["hx"])
        dz = _fp_witness(
            rng, _fp_columns(item["hx"], item["n"]),
            identity, item["zs"], trials=6, best=bound,
            descent_rows=item["hz"])
        item["full"] = min(dx, dz)

    winner = max(finalists, key=merit)
    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 275*winner["n"] <= 1000*winner["k"] < 325*winner["n"]
    return {"idx": idx,
            "cells": {"A": winner["A"], "B": winner["B"]}}


from functools import lru_cache


def _affine_signature(hx, hz, n):
    """Permutation-invariant combined-Tanner walk fingerprint.

    Ignore X/Z colouring as well as all group/protograph labels.
    Different fingerprints exclude incidence-graph isomorphism;
    equal fingerprints conservatively discard possible false duplicates.
    """
    from collections import Counter
    rows = hx + hz
    columns = _fp_columns(rows, n)
    adjacency = [{} for _ in rows]
    for syndrome in columns:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for t, a in enumerate(neighbors):
            for b in neighbors[:t]:
                adjacency[a][b] = adjacency[a].get(b, 0) + 1
                adjacency[b][a] = adjacency[b].get(a, 0) + 1

    profiles = []
    for i, neighbors in enumerate(adjacency):
        square = {}
        for j, multiplicity in neighbors.items():
            for h, other in adjacency[j].items():
                square[h] = square.get(h, 0) + multiplicity * other
        profiles.append((
            rows[i].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(i, 0),
            sum(value * neighbors.get(h, 0)
                for h, value in square.items()),
            sum(value * value for value in square.values()),
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(Counter(profiles).items())),
    )


@lru_cache(maxsize=1)
def _affine_bank(G):
    """Build independently optimized, invariant-distinct card designs."""
    bank = []
    fingerprints = set()
    for slot in range(N_SLOTS):
        rng = random.Random(30075025 * 10007 + slot)
        selected = None
        for retry in range(4):
            try:
                ctx = _affine_optimize_ctx(rng, G, slot)
            except ValueError:
                continue
            A, B = ctx["cells"]["A"], ctx["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            fingerprint = _affine_signature(hx, hz, n)
            if fingerprint in fingerprints:
                continue
            assert css_ok(hx, hz)
            assert all(len(cell) == 3 and len(set(cell)) == 3
                       for grid in (A, B) for row in grid for cell in row)
            fingerprints.add(fingerprint)
            selected = (
                tuple(tuple(tuple(cell) for cell in row) for row in A),
                tuple(tuple(tuple(cell) for cell in row) for row in B),
            )
            break
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return one fresh card design, never a transported orbit copy."""
    selected = _affine_bank(G)[idx]
    if selected is None:
        raise ValueError("no new packed cubic design passed rate/logical screens")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: uniform labels with legal rolled counts alone exclude light logicals; the parent found many weight-2 witnesses and a measured top-five witness of weight 7.
# REFUTED: Four terms in every C37:C3 cell alone ensure a measured distance floor of 21; current-parent slot 13 has four terms throughout but d30s=18, with sector-one X minima.
# REFUTED: Fresh tall C37:C3 components plus transpose-kernel/full-logical descent necessarily exceed fitness 91.7; the current program scored 91.378 with five measured weight-21 sector-two minima.
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
