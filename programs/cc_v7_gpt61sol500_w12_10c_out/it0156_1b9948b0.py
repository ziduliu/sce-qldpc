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
CONCEPT = ("Dic10, order 40, A=(1,2), B=(2,6): balanced degree-three B columns across two check rows, with separately packed check-pair orbits, should raise five-code measured fitness above the 0.300 frontier's 32.5; this is a falsifiable hypothesis.")
GROUP = ("Retain the dicyclic presentation rule, resizing only its parameter from Dic21 to Dic10 because the new size factor fourteen excludes order 84. Order 40 is the largest dicyclic order fitting n<600 and gives n=560. The related Dic11 (1,2)x(2,5) bed reached fitness 37.0, motivating a second B check row without claiming distance transfer.")
SHAPES = ("Target 0.300 with (1,2)x(2,6): n=560, kb=160, guaranteed rate 2/7, component variable/check counts 80/40 and 240/80, and LP sectors of 480 and 80 qubits. Parent weight-ten X minima occupied sector-one Acol/Bcol blocks; measured slot 21 also retained a mixed-sector weight-nine witness touching Arow0/Brow0. Split B constraints across two rows and screen both sectors. The shorter A component is a risk, not an assumed improvement. Verified ranks retain [0.275,0.325).")
FILLING = ("Twenty cards cross five connected B support profiles, intrinsically distinguished by zero through four pure-row columns, with four A rotation-pair-order profiles. Every component column has degree three; A's row sum is six and B's row sums are nine, giving HX weight nine and HZ weight twelve. Pack directed ratios separately for each check-row pair, then verify component four/six-cycles, ranks and kernel witnesses using scaffold lifts. Full logical searches add original stabilizer rows as descent moves and explicitly seed nontrivial dependencies. Uncoloured combined-Tanner walk fingerprints exclude emitted orbit copies. Component readings are selection heuristics; full logical witness readings remain upper bounds, never distance certificates.")
FAMILY_NAME = "r030_dic10_12x26_balanced_cubic_pairpack_descent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    # Same dicyclic rule, resized to Dic10:
    # x^20=1, y^2=x^10, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic10 has order 40")
    m, half = 20, 10
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + half * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + half*b*d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + half) % m, 1)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


# Each tuple is a B-column's term counts in its two check rows.
# Every column totals three; both row totals are nine.
# Pure-row column counts 0,1,2,3,4 distinguish these support
# geometries under row/column permutations. Each has mixed columns,
# so none is a deliberately disconnected pair of one-row components.
_SHAPE_B_COLUMNS = (
    ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
    ((3,0), (2,1), (1,2), (1,2), (1,2), (1,2)),
    ((3,0), (0,3), (2,1), (2,1), (1,2), (1,2)),
    ((3,0), (3,0), (0,3), (1,2), (1,2), (1,2)),
    ((3,0), (3,0), (0,3), (0,3), (2,1), (1,2)),
)

# gcd(step,20)=1,2,4 gives rotation-pair orders 20,10,5.
# These prescribe internal relative structure, not absolute anchors.
_SHAPE_A_PAIRS = ((1,1), (1,2), (2,2), (1,4))


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


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
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
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _component_data(hx, hz, q):
    """Extract every component row from the VERIFIED LP construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]  # Fix B-column zero.
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bm = (1 << (cb*q)) - 1
    # Fix A-column zero; keep ALL B check rows.
    br = [row & bm for row in hz[:rb*q]]
    return ar, br, _columns(ar, ca*q), _columns(br, cb*q)


def _shape_ctx(rng, G, idx):
    """Optimize this support/order card using only verified matrix screens."""
    from math import gcd
    assert G.q == 40
    card = idx % 20
    geometry, mode = divmod(card, 4)
    ad = _SHAPE_A_PAIRS[mode]
    columns = _SHAPE_B_COLUMNS[geometry]
    sa = [[3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    m = G.q // 2
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    steps = {
        divisor: [d for d in range(1, m) if gcd(d, m) == divisor]
        for divisor in (1, 2, 4)
    }

    def draw_cell(weight, divisor):
        if not weight:
            return []
        first = G.pick(rng)
        if weight == 1:
            return [first]
        b = first[1]
        second = ((first[0] + rng.choice(steps[divisor])) % m, b)
        cell = [first, second]
        if weight == 3:
            cell.append(rng.choice(pools[1-b]))
        assert len(cell) == weight and len(set(cell)) == weight
        return cell

    def packed(side):
        """Pack each ordered pair of component check-row blocks separately.

        For A's left action the invariant is r1*r2^-1; for B's
        right action it is r1^-1*r2. Both give label ratios g*h^-1.
        This is only a proposal filter: verified matrices subsequently
        check the actual four-cycle count.
        """
        counts = sa if side == "A" else sb
        rows, width = len(counts), len(counts[0])
        for restart in range(24):
            used = {(i, h): set() for i in range(rows)
                    for h in range(rows)}
            grid = [[[] for _ in range(width)] for _ in range(rows)]
            order = list(range(width))
            rng.shuffle(order)
            failed = False
            for j in order:
                accepted = False
                for attempt in range(192):
                    column = []
                    for i in range(rows):
                        weight = counts[i][j]
                        # B binomials use long rotation pairs. Pure B
                        # triples explore the card's prescribed orders.
                        divisor = (ad[j] if side == "A" else
                                   (1 if weight == 2 else
                                    (1, 2, 4)[(mode+i+j) % 3]))
                        column.append(draw_cell(weight, divisor))
                    additions = {}
                    legal = True
                    for i in range(rows):
                        for h in range(rows):
                            pairs = [
                                G._key(G.mul(g, G.inv(t)))
                                for g in column[i] for t in column[h]
                                if i != h or g != t
                            ]
                            ratios = set(pairs)
                            if (len(ratios) != len(pairs)
                                    or ratios & used[(i, h)]):
                                legal = False
                                break
                            additions[(i, h)] = ratios
                        if not legal:
                            break
                    if not legal:
                        continue
                    for key, ratios in additions.items():
                        used[key].update(ratios)
                    for i in range(rows):
                        grid[i][j] = column[i]
                    accepted = True
                    break
                if not accepted:
                    failed = True
                    break
            if not failed:
                return grid
        return None

    def six_cycles(cols):
        """Exact Tanner six-cycle count for a four-cycle-free cubic matrix.

        Each cubic variable contributes one same-variable triangle in the
        check adjacency graph; subtract those from its triangle count.
        """
        height = max((s.bit_length() for s in cols), default=0)
        adjacency = [0] * height
        for syndrome in cols:
            neighbors = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbors.append(bit.bit_length() - 1)
                syndrome ^= bit
            assert len(neighbors) == 3
            for a in neighbors:
                for b in neighbors:
                    if a != b:
                        adjacency[a] |= 1 << b
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            higher = neighbors >> (a + 1)
            while higher:
                bit = higher & -higher
                b = a + 1 + bit.bit_length() - 1
                triangles += (
                    (neighbors & adjacency[b]) >> (b + 1)
                ).bit_count()
                higher ^= bit
        return triangles - len(cols)

    def row_basis(rows):
        basis = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        return basis

    def outside(word, basis):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def probe(cols, trials, start=None, stabilizers=None, descent=()):
        """Actual kernel witnesses with information-set recombination.

        For full-code searches, a witness lowers the reading only when it
        lies outside the matching stabilizer row space. Kernel descent
        may use stabilizers, but a stabilizer never becomes a logical.
        """
        best = len(cols) + 1 if start is None else start
        order = list(range(len(cols)))
        memory = []

        def consider(word):
            nonlocal best
            weight = word.bit_count()
            if not word or weight >= best:
                return
            if stabilizers is None or outside(word, stabilizers):
                best = weight

        for trial in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = []
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
                    words.append(word)
                    consider(word)

            words.sort(key=int.bit_count)
            discoveries = set(words[:40])
            discoveries.update(memory)
            cutoff = min(best + 4, len(cols))
            for i, word in enumerate(words):
                for other in words[:i]:
                    combined = word ^ other
                    if combined and combined.bit_count() <= cutoff:
                        consider(combined)
                        discoveries.add(combined)

            # Recombine short words from different information sets.
            short = sorted(discoveries, key=int.bit_count)[:48]
            for i, word in enumerate(short):
                for other in short[:i]:
                    combined = word ^ other
                    if combined and combined.bit_count() <= cutoff:
                        consider(combined)
                        discoveries.add(combined)

            # Original low-weight checks are especially useful for
            # exposing mixed-sector logical representatives.
            reducers = words + memory + list(descent)
            seeds = sorted(discoveries, key=int.bit_count)[:6]
            if stabilizers is not None:
                nontrivial = []
                for word in words:
                    if outside(word, stabilizers):
                        nontrivial.append(word)
                        if len(nontrivial) == 6:
                            break
                # Do not let short stabilizers monopolize descent seeds.
                seeds.extend(nontrivial)
            for seed in seeds:
                word = seed
                for step in range(3):
                    chosen, weight = word, word.bit_count()
                    for reducer in reducers:
                        combined = word ^ reducer
                        if combined and combined.bit_count() < weight:
                            chosen, weight = combined, combined.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                    consider(word)
                    discoveries.add(word)
            memory = sorted(discoveries, key=int.bit_count)[:48]
            if best <= 4:
                break
        return best

    def score(candidate):
        d = min(candidate["da"], candidate["db"], candidate["full"])
        return (candidate["k"] * d * d / candidate["n"], d,
                candidate["da"] + candidate["db"],
                -candidate["six"][1], -sum(candidate["six"]))

    # First optimize genuine graph structure. All retained matrices have
    # cubic columns and no component four-cycles, not merely high search
    # readings from a short information-set budget.
    raw = []
    for attempt in range(80):
        A, B = packed("A"), packed("B")
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br, ac, bc = _component_data(hx, hz, G.q)
        if _component_cycles(ar) or _component_cycles(br):
            continue
        ta, tb = six_cycles(ac), six_cycles(bc)
        structure = (2 * tb + ta, max(ta, tb), tb, ta)
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
            "ac": ac, "bc": bc, "six": (ta, tb),
            "structure": structure, "full": n + 1,
        }
        raw.append(candidate)
        raw.sort(key=lambda c: c["structure"])
        del raw[12:]

    pool = []
    for candidate in raw:
        hx, hz, n = candidate["hx"], candidate["hz"], candidate["n"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 275 * n <= 1000 * k < 325 * n:
            continue
        candidate["k"] = k
        if min(_short_floor(candidate["ac"]),
               _short_floor(candidate["bc"])) < 5:
            continue
        candidate["da"] = probe(candidate["ac"], 10)
        candidate["db"] = probe(candidate["bc"], 14)
        # Require actual component witnesses, rather than scoring a
        # no-witness sentinel as a large distance.
        if (candidate["da"] > len(candidate["ac"])
                or candidate["db"] > len(candidate["bc"])):
            continue
        pool.append(candidate)

    if not pool:
        raise ValueError("no rank-legal packed cubic candidate")

    finalists = sorted(pool, key=score, reverse=True)[:4]
    for candidate in finalists:
        candidate["da"] = probe(
            candidate["ac"], 32, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 48, candidate["db"])
        candidate["xs"] = row_basis(candidate["hx"])
        candidate["zs"] = row_basis(candidate["hz"])
        candidate["xc"] = _columns(candidate["hz"], candidate["n"])
        candidate["zc"] = _columns(candidate["hx"], candidate["n"])
        # Include both sectors and shorten with actual stabilizer rows.
        # Start with a no-witness sentinel, not a component reading:
        # only verified non-stabilizer full-code words lower this value.
        dx = probe(candidate["xc"], 14, None, candidate["xs"],
                   descent=candidate["hx"])
        dz = probe(candidate["zc"], 14, None, candidate["zs"],
                   descent=candidate["hz"])
        candidate["full"] = min(dx, dz)

    leaders = sorted(finalists, key=score, reverse=True)[:2]
    for candidate in leaders:
        candidate["da"] = probe(
            candidate["ac"], 48, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 64, candidate["db"])
        dx = probe(candidate["xc"], 24, None, candidate["xs"],
                   descent=candidate["hx"])
        dz = probe(candidate["zc"], 24, None, candidate["zs"],
                   descent=candidate["hz"])
        candidate["full"] = min(candidate["full"], dx, dz)

    winner = max(finalists, key=score)
    assert css_ok(winner["hx"], winner["hz"])
    assert winner["n"] == 560
    assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
    assert max(row.bit_count()
               for row in winner["hx"] + winner["hz"]) <= W_MAX
    for M, counts in ((winner["A"], sa), (winner["B"], sb)):
        assert all(len(cell) == counts[i][j]
                   and len(set(cell)) == len(cell) <= 5
                   for i, row in enumerate(M)
                   for j, cell in enumerate(row))
    return {"idx": idx, "card": card,
            "cells": {"A": winner["A"], "B": winner["B"]},
            "screen": score(winner), "six": winner["six"]}


def _shape_fingerprint(hx, hz, n):
    """Invariant under arbitrary permutations of qubits and all checks.

    Ignore X/Z colouring and every group/protograph label. Different
    fingerprints exclude Tanner isomorphism; equal fingerprints
    conservatively reject possible false duplicates.
    """
    from collections import Counter
    rows = hx + hz
    columns = _columns(rows, n)
    adjacency = [Counter() for _ in rows]
    for syndrome in columns:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for t, a in enumerate(checks):
            for b in checks[:t]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1
    profiles = []
    for a, neighbors in enumerate(adjacency):
        square = {}
        for b, multiplicity in neighbors.items():
            for c, other in adjacency[b].items():
                square[c] = square.get(c, 0) + multiplicity * other
        profiles.append((
            rows[a].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(a, 0),
            sum(value * neighbors.get(c, 0)
                for c, value in square.items()),
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(Counter(profiles).items())),
    )


from functools import lru_cache


@lru_cache(maxsize=1)
def _shape_bank(G):
    """Pure deterministic optimization of invariant-distinct card designs."""
    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        local = random.Random(BASE_SEED * 10007 + 40260017 + 1009*slot)
        selected = None
        for retry in range(5):
            try:
                ctx = _shape_ctx(local, G, slot)
            except ValueError:
                continue
            A, B = ctx["cells"]["A"], ctx["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            fingerprint = _shape_fingerprint(hx, hz, n)
            if fingerprint in fingerprints:
                continue
            assert css_ok(hx, hz)
            fingerprints.add(fingerprint)
            selected = (
                tuple(tuple(tuple(cell) for cell in row) for row in A),
                tuple(tuple(tuple(cell) for cell in row) for row in B),
                ctx["screen"], ctx["six"],
            )
            break
        # A failed card is not replaced by an orbit copy or archived code.
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's independently optimized, incidence-distinct code."""
    selected = _shape_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh balanced-cubic target-rate design")
    A, B, screen, six = selected
    return {"idx": idx, "card": idx % 20,
            "cells": {"A": A, "B": B}, "screen": screen, "six": six}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Order-42 binomials preserve the cubic bed's distance advantage; every distinct binomial-profile parent slot read d2s=5, while all five measured codes used five trinomials.
# REFUTED: Component-kernel selection alone removes the parent's mixed-sector bottleneck; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Joint sampled component/full-code screening removes every mixed-sector weight-nine X failure; current-parent measured slot 21 still had d30s=9 with a witness touching Arow0/Brow0.
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
