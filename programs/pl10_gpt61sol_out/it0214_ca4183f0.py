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
CONCEPT = ("C9 semidirect C3 with action 4, order 27, shapes (1,4)x(1,4): noncommuting order-nine binomials and reduced short-logical hook exposure should improve the open-ended 0.550 frontier beyond 3.5010; measured worse-basis circuit failures decide this hypothesis.")
GROUP = ("Keep the metacyclic semidirect-product construction rule under L2, resizing its parameters because 17*42 exceeds the size window. Action 4 has order three modulo nine, giving an associative odd-order nonabelian group without involutions. This group/shape bed is untried. The C35 high-rate frontier scores 3.5010 with widespread weight-four logicals and frequent failures; that motivates a different internal geometry, not a demonstrated group advantage.")
SHAPES = ("Equal A=(1,4), B=(1,4) components have 108 variables and 27 checks each. At q=27, n=459 and kb=243 give guaranteed rate 9/17=0.529412, so every positive-rank realization belongs to the open-ended 0.550 cell. Parent failures spread across Acol/Bcol blocks with mostly 1.0-1.2x enrichment and collective flips; no persistent Acol/Brow hotspot is established. Equal component lengths replace the parent's 84/252-variable imbalance.")
FILLING = ("Both sides use [2,2,2,3]: component row totals nine, both CSS weight bounds twelve, and 14580 six-round CNOTs versus the parent's 25704 and the C35 frontier's 18900. Each side's three binomial directions have order nine and are pairwise noncommuting. Twenty indexed canonical geometry windows cross trinomial spectra (9,9,9) and (3,9,9), rejecting obvious gauge, automorphism and column-orbit copies before selection. Verified component actions shortlist by actual short-kernel collisions, rank deficits and cycles; full verified CSS matrices compare nontrivial logical triples/four-tuples and shared-opposite-check hook exposure. These are pL surrogates, not distance certificates. Parent slots 15 and 17 shared n=588,k=176,w=12,d2s=10 but scored 5.479 and 6.277, so reported distance and rate do not settle circuit reliability.")
FAMILY_NAME = "r055_C9byC3_s4_14x14_noncommuting_hook_geometry_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 27

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C9 semidirect C3 requires q=27")
    # Same metacyclic construction rule, with resized parameters.
    # x^9=y^3=1, y*x*y^-1=x^4; 4^3=1 modulo 9.
    m, r, s = 9, 3, 4
    act = tuple(pow(s, b, m) for b in range(r))
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + act[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-act[bi] * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _fg_columns(rows, width):
    """Column syndromes of a verified component action."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _fg_cost(rows, width):
    """Connectivity, exact rank deficit, and exact Tanner 4/6 cycles."""
    from itertools import combinations

    cols = _fg_columns(rows, width)
    if not all(cols) or not all(rows):
        return None
    neighbors = [[] for _ in range(width)]
    adj = [dict() for _ in rows]
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            neighbors[c].append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, b in combinations(neighbors[c], 2):
            adj[a][b] = adj[a].get(b, 0) + 1
            adj[b][a] = adj[b].get(a, 0) + 1

    seen, todo = {0}, [0]
    while todo:
        a = todo.pop()
        for b in adj[a]:
            if b not in seen:
                seen.add(b)
                todo.append(b)
    if len(seen) != len(rows):
        return None

    four = sum(v * (v - 1) // 2
               for a in range(len(rows))
               for b, v in adj[a].items() if a < b)
    six = 0
    for a in range(len(rows)):
        for b, ab in adj[a].items():
            if b <= a:
                continue
            for c, bc in adj[b].items():
                if c > b and c in adj[a]:
                    six += ab * bc * adj[a][c]
    for ns in neighbors:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return len(rows) - gf2_rank(rows), four, six


def _fg_quality(rng, rows, width, q):
    """Exact kernel<=4 exclusion and small-boundary/kernel heuristics.

    Triple boundaries are enumerated, but triple syndromes are not stored:
    this is not an exact exclusion test through weight six.
    Randomized kernel circuits give upper bounds, never pL predictions.
    """
    cols = _fg_columns(rows, width)
    seen = {0}
    for value in cols:
        if value in seen:
            return True, 4, 1.0e9, 1.0e9
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            if syndrome in seen:
                return True, 4, 1.0e9, 1.0e9
            seen.add(syndrome)

    exposure = 0
    loads = [0] * (width // q)
    for i in range(width):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                weight = (pair ^ cols[h]).bit_count()
                if weight < 5:
                    penalty = (5 - weight) ** 2
                    exposure += penalty
                    loads[i // q] += penalty
                    loads[j // q] += penalty
                    loads[h // q] += penalty
    triples = width * (width - 1) * (width - 2) // 6
    boundary = exposure / max(1, triples)
    peak = max(loads) * len(loads) / max(1, 3 * triples)

    order = list(range(width))
    best = width + 1
    for _ in range(28):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = syndrome, word
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                best = min(best, word.bit_count())
                for previous in basis:
                    best = min(best, (word ^ previous).bit_count())
                basis.append(word)
        if best <= 6:
            break
    return False, best, boundary, peak


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def _hr_hook_profile(checks, stabilizers, n):
    """Exact small-logical collisions with a shared-check hook surrogate.

    Quotient labels distinguish stabilizers from logical differences.
    Shared opposite checks do not certify the actual ASC hook ordering.
    """
    syndrome = _fg_columns(checks, n)
    if 0 in syndrome or len(set(syndrome)) != n:
        return None
    spread = _fg_columns(stabilizers, n)
    pivots = _row_span(stabilizers)
    pivot_order = sorted(pivots, reverse=True)
    residual = []
    for c in range(n):
        word = 1 << c
        for p in pivot_order:
            if (word >> p) & 1:
                word ^= pivots[p]
        residual.append(word)
    singles = dict(zip(syndrome, residual))
    pairs = {}
    three = four = double_hook = hook_exposure = 0
    for i in range(n):
        si, ri, hi = syndrome[i], residual[i], spread[i]
        for j in range(i):
            value = si ^ syndrome[j]
            label = ri ^ residual[j]
            third = singles.get(value)
            if third is not None and label != third:
                three += 1
            hook = (hi & spread[j]).bit_count()
            old = pairs.get(value)
            if old is None:
                pairs[value] = [1, hook, {label: [1, hook]}]
                continue
            same = old[2].get(label)
            same_count, same_hooks = same if same else (0, 0)
            different = old[0] - same_count
            different_hooks = old[1] - same_hooks
            four += different
            double_hook += hook * different_hooks
            hook_exposure += hook * different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook
    return three // 3, double_hook, hook_exposure, four // 3


def NEW_CTX(rng, G, idx):
    """Indexed inequivalent geometry windows, screened on verified lifts."""
    q = G.q
    assert q == 27 and A_SHAPE == B_SHAPE == (1, 4)
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert sum(counts[0]) + max(counts[0]) == 12
    assert _grid_weights(counts, counts) <= W_MAX

    design, mode = idx % 20, idx // 20
    pair_types = ((0, 0), (0, 1), (1, 1))
    if design < 8:
        pair_number, card = 0, design + 8 * mode
    elif design < 15:
        pair_number, card = 1, design - 8 + 7 * mode
    else:
        pair_number, card = 2, design - 15 + 5 * mode
    ta, tb = pair_types[pair_number]
    elements = list(G.elements)

    def powers(g, length):
        out = [G.e]
        for _ in range(1, length):
            out.append(G.mul(out[-1], g))
        return out

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: element_order(g) for g in elements}
    representatives = [
        g for g in elements
        if orders[g] == 9 and G.idx(g) < G.idx(G.inv(g))
    ]
    # Order-nine commuting classes are three distinct projective directions.
    lines = []
    remaining = list(representatives)
    while remaining:
        u = remaining[0]
        line = [v for v in remaining if G.mul(u, v) == G.mul(v, u)]
        lines.append(line)
        remaining = [v for v in remaining if v not in line]
    assert len(lines) == 3 and all(len(line) == 3 for line in lines)

    spectra = ((9, 9, 9), (3, 9, 9))
    triangle_books = [[], []]
    usable = [g for g in elements if orders[g] in (3, 9)]
    for i, u in enumerate(usable):
        for v in usable[i + 1:]:
            if G.mul(u, v) == G.mul(v, u):
                continue
            spectrum = tuple(sorted((
                orders[u], orders[v], orders[G.mul(G.inv(u), v)])))
            if spectrum in spectra:
                triangle_books[spectra.index(spectrum)].append((u, v))
    assert all(triangle_books)

    # Enumerate actual automorphisms of this presentation, not guessed
    # coordinate multipliers. Generator images must satisfy the relation
    # and induce a bijection on all 27 elements.
    automorphisms = set()
    for u in elements:
        if orders[u] != 9:
            continue
        up = powers(u, 9)
        for v in elements:
            if orders[v] != 3:
                continue
            if G.mul(G.mul(v, u), G.inv(v)) != up[4]:
                continue
            vp = powers(v, 3)
            image = tuple(G.idx(G.mul(up[a], vp[b]))
                          for a, b in elements)
            if len(set(image)) == q:
                automorphisms.add(image)
    automorphisms = sorted(automorphisms)
    assert automorphisms

    image_cache = {}

    def cell_images(cell, side):
        signature = tuple(sorted(G.idx(g) for g in cell))
        key = side, signature
        if key not in image_cache:
            images = []
            for permutation in automorphisms:
                mapped = [elements[permutation[t]] for t in signature]
                normalized = []
                for anchor in mapped:
                    ai = G.inv(anchor)
                    if side == 0:
                        values = [G.mul(g, ai) for g in mapped]
                    else:
                        values = [G.mul(ai, g) for g in mapped]
                    normalized.append(tuple(sorted(G.idx(g) for g in values)))
                images.append(min(normalized))
            image_cache[key] = tuple(images)
        return image_cache[key]

    def canonical(A, B):
        """Remove obvious cell gauges, automorphisms and column images.

        Side exchange with element inversion is included. This is not
        a certificate against arbitrary Tanner graph isomorphisms.
        """
        aa = [cell_images(cell, 0) for cell in A[0]]
        bb = [cell_images(cell, 1) for cell in B[0]]
        exchanged_a = [
            cell_images([G.inv(g) for g in cell], 0) for cell in B[0]]
        exchanged_b = [
            cell_images([G.inv(g) for g in cell], 1) for cell in A[0]]
        candidates = []
        for t in range(len(automorphisms)):
            candidates.append((
                tuple(sorted(image[t] for image in aa)),
                tuple(sorted(image[t] for image in bb))))
            candidates.append((
                tuple(sorted(image[t] for image in exchanged_a)),
                tuple(sorted(image[t] for image in exchanged_b))))
        return min(candidates)

    grng = random.Random(930271 + 1009 * pair_number)

    def draw(side_type):
        bins = [[G.e, grng.choice(line)] for line in lines]
        u, v = grng.choice(triangle_books[side_type])
        return [bins + [[G.e, u, v]]]

    # Exact singleton actions are cached from the scaffold helper.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, word in enumerate(actions[side][g]):
                    rows[r] ^= word << (j * q)
        return rows

    def short_counts(rows):
        """Kernel collision counts through four; zero certifies exclusion."""
        cols = _fg_columns(rows, 4 * q)
        singles = {}
        one = two = three = four = 0
        for value in cols:
            one += int(value == 0)
            two += singles.get(value, 0)
            singles[value] = singles.get(value, 0) + 1
        pairs = {}
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                three += singles.get(syndrome, 0)
                four += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1
        return one, two, three, four

    # Disjoint canonical windows are deliberate internal-geometry briefs.
    # Additional harness batches advance the book rather than recycle it.
    window = 24
    start, stop = card * window, (card + 1) * window
    seen = set()
    pool = []
    accepted = 0
    for trial in range(2000 + 400 * stop):
        A, B = draw(ta), draw(tb)
        identity = canonical(A, B)
        if identity in seen:
            continue
        seen.add(identity)
        serial = accepted
        accepted += 1
        if serial >= start:
            ar, br = component(A, 0), component(B, 1)
            ac = _fg_cost(ar, 4 * q)
            bc = _fg_cost(br, 4 * q)
            if ac is not None and bc is not None:
                ash, bsh = short_counts(ar), short_counts(br)
                coarse = (
                    tuple(max(a, b) for a, b in zip(ash[:3], bsh[:3])),
                    tuple(a + b for a, b in zip(ash[:3], bsh[:3])),
                    max(ash[3], bsh[3]), ash[3] + bsh[3],
                    ac[0] + bc[0], max(ac[1], bc[1]),
                    ac[1] + bc[1], ac[2] + bc[2], serial)
                pool.append((coarse, A, B, ar, br, identity))
        if accepted >= stop:
            break
    if accepted < stop or not pool:
        raise ValueError("canonical high-rate geometry window exhausted")
    pool.sort(key=lambda item: item[0])

    best = None
    for coarse, A, B, ar, br, identity in pool[:8]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 459 and k >= 9 * q and 40 * k > 21 * n):
            continue
        if not _connected_checks(hx, hz, n):
            continue
        actual_a = [
            sum(((hx[r] >> (j * 4 * q)) & mask) << (j * q)
                for j in range(4))
            for r in range(q)]
        actual_b = [row & ((1 << (4 * q)) - 1) for row in hz[:q]]
        assert actual_a == ar and actual_b == br

        px = _hr_hook_profile(hz, hx, n)
        pz = _hr_hook_profile(hx, hz, n)
        if px is None or pz is None:
            # Keep a legal code as a fallback, without claiming protection.
            profile_key = (1,) + (10**9,) * 8
        else:
            profile_key = (0,) + tuple(
                value for a, b in zip(px, pz)
                for value in (max(a, b), a + b))
        key = profile_key + (k - 9 * q, coarse)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, identity, px, pz)

    if best is None:
        raise ValueError("no connected verified high-rate geometry")
    key, A, B, hx, hz, k, identity, px, pz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert 6 * sum(row.bit_count() for row in hx + hz) == 14580
    return {"A": A, "B": B, "design": design, "mode": mode,
            "k": k, "screen": key, "geometry": identity,
            "contract": (spectra[ta], spectra[tb], card),
            "hook_profile": (px, pz)}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; no subsequent redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance alone raises measured fitness: zero-failure parent slot58 had d2s=22,k=45,s=7.165, below zero-failure slot20 with d2s=20,k=58,s=7.276; their sample ceilings limited both.
# REFUTED: Equal n,k,w,d2s certify equal circuit reliability: current-parent slots15/17 both reported 588,176,12,10, but scored 5.479/6.277; slot15 Z reached 100 failures after 34750 shots while slot17's worse X basis had 91 failures in 200000 shots.
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
