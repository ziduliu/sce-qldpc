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
CONCEPT = ("For C5 semidirect C4 with action 2, order 20, shapes (1,3)x(2,7), shared-cokernel A trinomials and B components selected against exact weight-five/six kernel words should produce five measured d30s>=6 readings in cell 0.500.")
GROUP = ("Preserve the metacyclic construction, normal C5 subgroup, and action 2; reduce only the complement from C8 to C4. Factor 23 excludes q=40. With m=5 and action of order four, q=20 is the largest fitting complement-order multiple of four. The supplied run state does not list this group/shape bed; the parent's five distance-five readings motivate changing the component architecture.")
SHAPES = ("A=(1,3), B=(2,7): n=460, kb=200 and guaranteed rate 10/23=0.43478. Components have 60 variables/20 checks and 140 variables/40 checks. Full-rank B with A deficits four or eight predicts k=220 or 240, both in cell 0.500; only exact assembled ranks decide acceptance. Parent weight-five X minima occupied sector 1, especially Acol1/Bcol2, while Z minima were at six.")
FILLING = ("Twenty cards cross five B support architectures with four quotient-occupancy models. A uses three incidence-distinct singular trinomials with common row dependencies. B has four cubic and three quadratic columns, row totals nine and column totals at most three; HX=HZ=12. Mandatory component exclusion through four precedes exact weight-five/six kernel censuses. Prefer the resulting component floor before rank surplus, then fewer short words and cycles. Full stabilizer-aware logical-four exclusion and combined uncoloured incidence fingerprints remain mandatory. Component floors are selection heuristics, not LP quantum-distance certificates.")
FAMILY_NAME = "r050_C5sdC4_action2_13x27_shared_cokernel_kernel6_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same metacyclic rule: C5 semidirect C4, action 2.
    # Only the complement order changes; 2^4=1 mod 5.
    if not valid_q(q):
        raise ValueError("C5 semidirect C4 requires order 20")
    m, r, s = 5, 4, 2
    els = [(a, b) for b in range(r) for a in range(m)]
    action = [pow(s, b, m) for b in range(r)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + action[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-action[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return a + m * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _four_cycle_count(rows):
    """Four-cycles in one Tanner graph, using the verified bit rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _has_small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight 1..4.

    Enumerate supports of size at most two. Equal syndromes yield
    every commuting support of size at most four. Test differences
    against the opposite stabilizer row space, not merely the kernel.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    syndrome = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndrome[low.bit_length() - 1] ^= bit
            row ^= low

    seen = {0: 0}
    for i, s in enumerate(syndrome):
        v = 1 << i
        old = seen.get(s)
        if old is not None:
            if outside(v ^ old):
                return True
        else:
            seen[s] = v

    for i in range(n):
        vi = 1 << i
        si = syndrome[i]
        for j in range(i):
            s = si ^ syndrome[j]
            v = vi | (1 << j)
            old = seen.get(s)
            if old is not None:
                if outside(v ^ old):
                    return True
            else:
                seen[s] = v
    return False


# Four cubic and three quadratic columns; every profile has row
# totals (9,9). These alter support architecture, not just labels.
_HALF_RATE_B_PROFILES = (
    ((2, 1), (2, 1), (1, 2), (1, 2),
     (1, 1), (1, 1), (1, 1)),
    ((3, 0), (0, 3), (2, 1), (1, 2),
     (1, 1), (1, 1), (1, 1)),
    ((2, 1), (2, 1), (1, 2), (1, 2),
     (2, 0), (0, 2), (1, 1)),
    ((2, 1), (2, 1), (2, 1), (1, 2),
     (0, 2), (1, 1), (1, 1)),
    ((3, 0), (0, 3), (2, 1), (1, 2),
     (2, 0), (0, 2), (1, 1)),
)


def _component_word_counts(rows, width):
    """Exact weight-five/six kernel census after exclusion through four.

    Distinct pair syndromes follow from the prior exclusion. A matching
    triple/pair has disjoint supports and represents a weight-five word.
    Matching distinct triples have disjoint supports and represent six.
    Each word contributes exactly ten unordered partitions.
    """
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    assert 0 not in cols and len(set(cols)) == width
    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            pairs.add(x ^ cols[j])
    assert len(pairs) == width * (width - 1) // 2

    triples = {}
    five = six = 0
    odd_columns = all(c.bit_count() & 1 for c in cols)
    for i in range(2, width):
        for j in range(1, i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if not odd_columns and syndrome in pairs:
                    five += 1
                previous = triples.get(syndrome, 0)
                six += previous
                triples[syndrome] = previous + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


@__import__("functools").lru_cache(maxsize=1)
def _shared_cokernel_bank(G):
    """Optimize twenty support/occupancy cards using verified components."""
    from collections import Counter

    q = G.q
    assert q == 20
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dummy_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]

    def component_rows(cells, side):
        if side == "A":
            hx, hz, n = lifted_matrices(G, cells, dummy_b)
            rows = []
            for i in range(ra):
                for t in range(q):
                    row = hx[(i * cb) * q + t]
                    rows.append(sum(
                        ((row >> (cb*j*q)) & mask) << (j*q)
                        for j in range(ca)))
            return rows, ca*q
        hx, hz, n = lifted_matrices(G, dummy_a, cells)
        bm = (1 << (cb*q)) - 1
        return [row & bm for row in hz[:rb*q]], cb*q

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def dependency_space(rows):
        """Canonical exact space of dependencies between verified rows."""
        pivots, dependencies = {}, []
        for r, original in enumerate(rows):
            row, word = original, 1 << r
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (row, word)
                    break
                row ^= old[0]
                word ^= old[1]
            if not row:
                dependencies.append(word)

        basis = {}
        for word in dependencies:
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def fingerprint(hx, hz, n):
        """Permutation-invariant combined, uncoloured incidence signature.

        Different signatures imply different incidence graphs. Equal
        signatures are conservatively rejected, not asserted isomorphic.
        """
        rows = list(hx) + list(hz)
        adj = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adj[i][j] = overlap
                    adj[j][i] = overlap
        profiles = []
        for r, neighbors in enumerate(adj):
            items = list(neighbors.items())
            triangles = 0
            for a, (s, ws) in enumerate(items):
                for t, wt in items[:a]:
                    triangles += 2*ws*wt*adj[s].get(t, 0)
            profiles.append((
                rows[r].bit_count(),
                tuple(sorted(Counter(neighbors.values()).items())),
                sum(w*w for w in neighbors.values()),
                triangles))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(degrees).items())))

    # A single-block matrix is extracted from a full, verified LP lift.
    # Right translations merely permute its columns and are removed by
    # the column-incidence signature before blocks enter a class.
    classes = {}
    gauges = list(G.elements)
    for gauge in gauges:
        for a in range(1, q):
            for b in range(a + 1, q):
                cell = tuple(G.mul(gauge, g)
                             for g in (G.e, G.elements[a], G.elements[b]))
                A = [[list(cell), [G.e], [G.e]]]
                hx, hz, n = lifted_matrices(G, A, dummy_b)
                rows = [row & mask for row in hx[:q]]
                deficit = q - gf2_rank(rows)
                if deficit not in (4, 8):
                    continue
                if _has_small_logical(rows, [], q):
                    continue
                space = dependency_space(rows)
                incidence = tuple(sorted(columns(rows, q)))
                bucket = classes.setdefault(space, {})
                if incidence not in bucket and len(bucket) < 32:
                    bucket[incidence] = cell
        if sum(len(bucket) >= 3 for bucket in classes.values()) >= 24:
            break

    library = tuple(
        (space, tuple(bucket.values()))
        for space, bucket in classes.items() if len(bucket) >= 3)
    library = tuple(sorted(
        library, key=lambda item: (len(item[0]), item[0])))

    b_models = ((0, 0, 1), (0, 1, 2), (0, 1, 3), (0, 2, 3))
    target_deficits = (4, 8, 4, 8, 4)
    target_spreads = (6, 6, 9, 9, 3)
    fingerprints, bank = set(), []

    def triple(rng, model):
        used, cell = set(), []
        phase = rng.randrange(4)
        for offset in model:
            b = (phase + offset) % 4
            choices = [(a, b) for a in range(5) if (a, b) not in used]
            g = rng.choice(choices)
            used.add(g)
            cell.append(g)
        base = G.pick(rng)
        return [G.mul(base, g) for g in cell]

    def draw_b(rng, policy, mode):
        counts = _HALF_RATE_B_PROFILES[policy]
        B = [[None] * cb for _ in range(rb)]
        mono = [j for j, pair in enumerate(counts) if pair == (1, 1)]
        # Distinct relative voltages obstruct parallel degree-two columns.
        relatives = rng.sample(G.elements, len(mono))
        for j, relative in zip(mono, relatives):
            B[0][j], B[1][j] = [G.e], [G.inv(relative)]
        for j, pair in enumerate(counts):
            if pair == (1, 1):
                continue
            for i, count in enumerate(pair):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    rotation = mode == 0 or (
                        mode == 3 and (i + j) % 2 == 0)
                    step = ((rng.randrange(1, 5), 0) if rotation
                            else (rng.randrange(5), rng.choice((1, 3))))
                    base = G.pick(rng)
                    cell = [base, G.mul(base, step)]
                else:
                    cell = triple(rng, b_models[(mode + i) % 4])
                assert len(cell) == len(set(cell))
                B[i][j] = cell
        return B

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    for slot in range(N_SLOTS):
        card = slot % 20
        policy, mode = divmod(card, 4)
        rng = random.Random(20523719 + 100003*slot)
        winner = None

        # Rank deficit and quotient diversity are prescribed policies;
        # independent labels are optimized inside each policy.
        eligible = [
            item for item in library
            if len(item[0]) == target_deficits[policy]]
        if not eligible:
            eligible = list(library)

        # Exhausted cards are voided, never replaced with a low-rate
        # binomial fallback or a transported copy of another design.
        for fallback in (False,):
            if not eligible:
                continue
            sa = [[3, 3, 3]]
            counts = _HALF_RATE_B_PROFILES[policy]
            sb = [[pair[i] for pair in counts] for i in range(rb)]
            assert _grid_weights(sa, sb) == 12
            assert _no_dead_lines(sa, sb)

            apool, aseen = [], set()
            for attempt in range(160):
                space, cells = rng.choice(eligible)
                A = [[list(cell) for cell in rng.sample(cells, 3)]]
                sig = signature(A, dummy_b)
                if sig in aseen:
                    continue
                aseen.add(sig)
                ar, width = component_rows(A, "A")
                if _has_small_logical(ar, [], width):
                    continue
                deficit = q - gf2_rank(ar)
                spread = sum(len({g[1] for g in cell}) for cell in A[0])
                key = (deficit,
                       -abs(spread - target_spreads[policy]),
                       -_four_cycle_count(ar))
                apool.append((key, A))
                apool.sort(key=lambda item: item[0], reverse=True)
                del apool[6:]
                if len(apool) >= 6 and attempt >= 48:
                    break
            if not apool:
                continue

            bpool, bseen = [], set()
            for attempt in range(192):
                B = draw_b(rng, policy, mode)
                sig = signature(dummy_a, B)
                if sig in bseen:
                    continue
                bseen.add(sig)
                br, width = component_rows(B, "B")
                if _has_small_logical(br, [], width):
                    continue
                key = (_four_cycle_count(br), -gf2_rank(br))
                bpool.append((key, B))
                bpool.sort(key=lambda item: item[0])
                del bpool[6:]
                if len(bpool) >= 6 and attempt >= 48:
                    break
            if not bpool:
                continue

            # Deepen only the retained component pools. These censuses
            # use extracted verified matrices and enumerate every triple.
            deep_a, deep_b = [], []
            for ak, A in apool:
                ar, width = component_rows(A, "A")
                words = _component_word_counts(ar, width)
                assert words[0] == 0  # Every A column has odd degree.
                deep_a.append((ak, A, words))
            for bk, B in bpool:
                br, width = component_rows(B, "B")
                deep_b.append((bk, B,
                               _component_word_counts(br, width)))

            proposals = []
            for ak, A, aw in deep_a:
                for bk, B, bw in deep_b:
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (475*n <= 1000*k < 525*n):
                        continue
                    cycles = -ak[2] + bk[0]
                    # Exact COMPONENT floors, not quantum certificates:
                    # odd A degree also excludes every odd kernel word.
                    afloor = 6 if aw[1] else 8
                    bfloor = 5 if bw[0] else (6 if bw[1] else 7)
                    floor = min(afloor, bfloor)
                    key = (k * floor * floor, floor, -bw[0],
                           -(aw[1] + bw[1]), k, -cycles, ak[1])
                    proposals.append((key, A, B, hx, hz, n))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, A, B, hx, hz, n in proposals[:12]:
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in fingerprints:
                    continue
                assert css_ok(hx, hz), "verified lift failed CSS commutation"
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert _grid_weights(
                    [[len(cell) for cell in row] for row in A],
                    [[len(cell) for cell in row] for row in B]) <= 12
                winner = (A, B, k, fp, fallback)
                break
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B, k, fp, fallback = winner
        fingerprints.add(fp)
        bank.append({
            "cells": {"A": A, "B": B}, "k": k,
            "card": card, "fallback": fallback,
            "component_selection": key,
            "screen_no_le4": True})

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh support-card design screened at the exact target rate."""
    selected = _shared_cokernel_bank(G)[idx]
    if selected is None:
        raise ValueError("no incidence-distinct rank-legal filling passed logical-4 screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel rank surplus removes the C5:C8 (1,3)x(1,4) distance-five bottleneck: all five measured rank-surplus leaders had k=252 and d30s=5, with X minima entirely in sector 1.
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
