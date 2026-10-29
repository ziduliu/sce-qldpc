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
CONCEPT = ("C7 semidirect C9, action 2, order 63, A=(1,2), B=(1,3): long-order noncommuting trinomials screened on the central-C3 invariant subspace should raise five-code fitness above 30.7648 in cell 0.300.")
GROUP = ("Jump from Dic21 to x^7=y^9=1, y*x*y^-1=x^2. Since 2^3=1 mod 7, this defines an order-63 group; odd order makes it genuinely different from Dic21 and excludes involutions. The order-nine quotient generator has central cube, unlike faithful C3 extensions with noncentral order-three elements. This bed is absent from BEDS; the strong C37:C3 bed motivates action-based exploration without claiming distance transfer.")
SHAPES = ("Keep (1,2)x(1,3): n=441, kb=126, guaranteed rate 2/7, with component variable/check counts 126/63 and 189/63. Parent weight-ten X witnesses occupied both A columns across the B columns; measured slot 21 retained a mixed-sector weight-nine witness touching Arow0/Brow0. Verified ranks retain [0.275,0.325), and full searches include both sectors.")
FILLING = ("Twenty cards prescribe noncommuting relative-order triangles drawn from orders 7,9,21. Three distinct terms per cell give HX weight nine and HZ weight twelve. Pack action-specific pair differences and verify component four-cycle exclusion. Fold verified syndromes over the central order-three subgroup, reject invariant kernel words of physical weight at most nine, and incorporate found folded witnesses into component selection. Stabilizer-tested full searches add stabilizer descent. Combined-Tanner fingerprints prevent emitted orbit copies. Readings remain upper bounds, not distance certificates.")
FAMILY_NAME = "r030_C7sdC9_action2_12x13_longtriangles_central3_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 63

def custom_ops(q):
    # x^7=y^9=1, y*x*y^-1=x^2.
    # The action has order three; y^3 is central of order three.
    if not valid_q(q):
        raise ValueError("C7 semidirect C9 requires order 63")
    els = [(a, b) for b in range(9) for a in range(7)]
    action = (1, 2, 4)

    def red(g):
        return (g[0] % 7, g[1] % 9)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b % 3]*c) % 7, (b+d) % 9)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 9
        return ((-action[bi % 3]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


from functools import lru_cache


@lru_cache(maxsize=1)
def _triangle_catalog(G):
    """Element recipes only; all matrix screens use verified lifts."""
    inverses = {g: G.inv(g) for g in G.elements}
    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, G.q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("invalid element order")

    catalog = {}
    legal = [g for g in G.elements if orders[g] >= 7]
    for r in legal:
        for s in legal:
            if G.mul(r, s) == G.mul(s, r):
                continue
            third = G.mul(inverses[r], s)
            signature = tuple(sorted(
                (orders[r], orders[s], orders[third])))
            if signature[0] < 7:
                continue
            cell = (G.e, r, s)
            differences = {
                G.mul(g, inverses[h])
                for g in cell for h in cell if g != h
            }
            if len(differences) == 6:
                catalog.setdefault(signature, []).append((r, s))
    return catalog


@lru_cache(maxsize=1)
def _central_orbits(G):
    """Partition by the central order-three translation, not a lift."""
    generator = (0, 3)
    assert G.mul(generator, G.mul(generator, generator)) == G.e
    assert all(G.mul(generator, g) == G.mul(g, generator)
               for g in G.elements)
    covered, partition = set(), []
    for g in G.elements:
        if G.idx(g) in covered:
            continue
        orbit, h = [], g
        for _ in range(3):
            orbit.append(G.idx(h))
            h = G.mul(h, generator)
        assert h == g and len(set(orbit)) == 3
        covered.update(orbit)
        partition.append(tuple(sorted(orbit)))
    assert len(covered) == G.q
    return tuple(partition)


def _central_fold(columns, G):
    """Exact restriction of VERIFIED syndromes to orbit-constant words.

    A compact kernel support of weight t expands to physical weight 3t.
    Central translation commutes with both component actions. Thus output
    syndromes are constant on output orbits, and one bit per orbit suffices.
    """
    orbits = _central_orbits(G)
    masks = [sum(1 << h for h in orbit) for orbit in orbits]
    folded = []
    for block in range(len(columns) // G.q):
        for orbit in orbits:
            syndrome = 0
            for h in orbit:
                syndrome ^= columns[block*G.q+h]
            compact = 0
            for r, mask in enumerate(masks):
                restricted = syndrome & mask
                assert restricted == 0 or restricted == mask
                if restricted:
                    compact |= 1 << r
            folded.append(compact)
    return folded


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
    """Extract both actions from the scaffold's verified LP construction."""
    mask = (1 << q) - 1
    # HX beta=0: the A blocks are sector-one blocks 0 and 3.
    ar = [
        (row & mask) | (((row >> (3*q)) & mask) << q)
        for row in hx[:q]
    ]
    # HZ A-column=0: B occupies the first three sector-one blocks.
    bm = (1 << (3*q)) - 1
    br = [row & bm for row in hz[:q]]
    return ar, br, _columns(ar, 2*q), _columns(br, 3*q)


def _optimize_ctx(rng, G, idx):
    """Optimize one relative-order architecture and its central restriction."""
    card = idx % 20
    catalog = _triangle_catalog(G)
    palettes = sorted(
        catalog, key=lambda s: (s[0], sum(s), s), reverse=True)[:3]
    if len(palettes) != 3:
        raise ValueError("three long-order triangle types required")
    a_cards = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2))
    b_cards = ((0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 2, 2))
    ad = tuple(palettes[t] for t in a_cards[card // 4])
    bd = tuple(palettes[t] for t in b_cards[card % 4])
    sa, sb = [[3, 3]], [[3, 3, 3]]
    # HX: 6+3=9. HZ: 9+3=12.
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def packed(signatures, side):
        """Pack the appropriate action's pair differences.

        This is only a proposal filter; verified matrix slices separately
        check four-cycles. Anchors are optimized, not emitted as orbit copies.
        """
        for restart in range(24):
            used = set()
            row = [None] * len(signatures)
            order = list(range(len(signatures)))
            rng.shuffle(order)
            for j in order:
                for attempt in range(192):
                    r, s = rng.choice(catalog[signatures[j]])
                    anchor = G.pick(rng)
                    cell = [anchor, G.mul(anchor, r),
                            G.mul(anchor, s)]
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
                    assert len(set(cell)) == 3
                    used.update(differences)
                    row[j] = cell
                    break
                if row[j] is None:
                    break
            if all(cell is not None for cell in row):
                return [row]
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

    def probe(cols, trials, start=None, stabilizers=None):
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

            # Adding stabilizers preserves the logical class and can reveal
            # lighter mixed-sector representatives missed by kernel pairs.
            reducers = words + memory + list(stabilizers or ())
            seeds = sorted(discoveries, key=int.bit_count)[:6]
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
        # Component bounds are conservative selection heuristics; they
        # are not asserted to be quantum logical witnesses.
        d = min(candidate["da"], candidate["db"], candidate["full"])
        return (candidate["k"] * d * d / candidate["n"], d,
                candidate["fold_clean"],
                candidate["da"] + candidate["db"],
                -candidate["six"][1], -sum(candidate["six"]))

    # First optimize genuine graph structure. All retained matrices have
    # cubic columns and no component four-cycles, not merely high search
    # readings from a short information-set budget.
    raw = []
    for attempt in range(48):
        A, B = packed(ad, "A"), packed(bd, "B")
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
        del raw[10:]

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
        af = _central_fold(candidate["ac"], G)
        bf = _central_fold(candidate["bc"], G)
        fa, fb = _short_floor(af), _short_floor(bf)
        # Exact exclusion of invariant component words through physical
        # weight nine. This does not exclude all quantum weight-nine words.
        if min(fa, fb) < 4:
            continue
        candidate["fold_clean"] = int(fa == 5) + int(fb == 5)
        fd_a = (fa if fa < 5 else _kernel_reading(af, rng, 24))
        fd_b = (fb if fb < 5 else _kernel_reading(bf, rng, 24))
        candidate["da"] = min(probe(candidate["ac"], 10), 3*fd_a)
        candidate["db"] = min(probe(candidate["bc"], 14), 3*fd_b)
        # Require actual component witnesses, rather than scoring a
        # no-witness sentinel as a large distance.
        if (candidate["da"] > len(candidate["ac"])
                or candidate["db"] > len(candidate["bc"])):
            continue
        pool.append(candidate)

    if not pool:
        raise ValueError("no rank-legal packed cubic candidate")

    finalists = sorted(pool, key=score, reverse=True)[:3]
    for candidate in finalists:
        candidate["da"] = probe(
            candidate["ac"], 32, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 48, candidate["db"])
        candidate["xs"] = row_basis(candidate["hx"])
        candidate["zs"] = row_basis(candidate["hz"])
        candidate["xc"] = _columns(candidate["hz"], candidate["n"])
        candidate["zc"] = _columns(candidate["hx"], candidate["n"])
        # Include both LP sectors. Low-weight stabilizers are explicitly
        # removed from the logical reading.
        # Do not initialize a logical reading with an untested component
        # bound. Full readings count only found non-stabilizer witnesses.
        dx = probe(candidate["xc"], 14, None, candidate["xs"])
        dz = probe(candidate["zc"], 14, None, candidate["zs"])
        candidate["full"] = min(dx, dz)

    leaders = sorted(finalists, key=score, reverse=True)[:2]
    for candidate in leaders:
        candidate["da"] = probe(
            candidate["ac"], 48, candidate["da"])
        candidate["db"] = probe(
            candidate["bc"], 64, candidate["db"])
        dx = probe(candidate["xc"], 24, None, candidate["xs"])
        dz = probe(candidate["zc"], 24, None, candidate["zs"])
        candidate["full"] = min(candidate["full"], dx, dz)

    winner = max(finalists, key=score)
    assert css_ok(winner["hx"], winner["hz"])
    assert max(row.bit_count()
               for row in winner["hx"] + winner["hz"]) <= W_MAX
    assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
    assert all(len(cell) == 3 and len(set(cell)) == 3
               for M in (winner["A"], winner["B"])
               for row in M for cell in row)
    return {"idx": idx, "card": card,
            "cells": {"A": winner["A"], "B": winner["B"]},
            "screen": score(winner), "six": winner["six"]}


def _tanner_signature(hx, hz, n):
    """Permutation-invariant fingerprint of the combined incidence graph.

    Forget check colouring and all group/protograph labels. Different
    fingerprints exclude Tanner isomorphism; equal ones conservatively
    discard possible false duplicates. No distance claim is involved.
    """
    from collections import Counter
    rows = hx + hz
    columns = _columns(rows, n)
    adjacency = [{} for _ in rows]
    for syndrome in columns:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length()-1)
            syndrome ^= bit
        for t, a in enumerate(neighbors):
            for b in neighbors[:t]:
                adjacency[a][b] = adjacency[a].get(b, 0)+1
                adjacency[b][a] = adjacency[b].get(a, 0)+1

    profiles = []
    for i, neighbors in enumerate(adjacency):
        square = {}
        for j, multiplicity in neighbors.items():
            for h, other in adjacency[j].items():
                square[h] = square.get(h, 0)+multiplicity*other
        profiles.append((
            rows[i].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(i, 0),
            sum(value*neighbors.get(h, 0)
                for h, value in square.items()),
            sum(value*value for value in square.values()),
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(Counter(profiles).items())),
    )


@lru_cache(maxsize=1)
def _central_design_bank(G):
    """Deterministically construct independently optimized fresh designs."""
    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        selected = None
        for retry in range(3):
            local = random.Random(63070931 + 1009*slot + 104729*retry)
            try:
                ctx = _optimize_ctx(local, G, slot)
            except ValueError:
                continue
            A, B = ctx["cells"]["A"], ctx["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            fingerprint = _tanner_signature(hx, hz, n)
            if fingerprint in fingerprints:
                continue
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert css_ok(hx, hz)
            assert 275*n <= 1000*k < 325*n
            assert _grid_weights(
                [[len(c) for c in row] for row in A],
                [[len(c) for c in row] for row in B]) <= W_MAX
            fingerprints.add(fingerprint)
            selected = (
                tuple(tuple(tuple(c) for c in row) for row in A),
                tuple(tuple(tuple(c) for c in row) for row in B),
            )
            break
        # A failed slot does not erase successful designs in other slots.
        bank.append(selected)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return one invariant-distinct card design, never an orbit copy."""
    selected = _central_design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh central-screened target-rate design")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected cubic cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Order-42 binomials preserve the cubic bed's distance advantage; every distinct binomial-profile parent slot read d2s=5, while all five measured codes used five trinomials.
# REFUTED: Component-kernel selection alone removes the parent's mixed-sector bottleneck; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Joint sampled component/full-code screening removes every mixed-sector weight-nine X failure on Dic21; current-parent measured slot 21 still had d30s=9 with a witness touching Arow0/Brow0.
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
