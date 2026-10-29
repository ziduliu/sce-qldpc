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
CONCEPT = ("Dicyclic Dic9, order 36, shapes (1,3)x(1,5): triangle-free all-binomial components, reduced extraction exposure and balanced gate-correlated decoding screening should improve the rate-0.500 frontier record of 3.2; measured circuit pL can refute this.")
GROUP = ("Jump from abelian C35 to nonabelian Dic9, equivalently C9 semidirect C4 with inversion. Relative orders 4,9,18 avoid involutive doubletons and order-three single-cell cycles. The order-36 (1,3)x(1,5) bed is absent from BEDS; neither this absence nor the algebra establishes a circuit advantage.")
SHAPES = ("Unequal single-row components have 108 variables/36 checks and 180 variables/36 checks. n=576 and kb=288 guarantee rate 0.500; final verified ranks require k=296, rate 0.5139 within target cell 0.500. Parent failures and weight-four witnesses were diffuse across Acol/Bcol blocks with about 1.0-1.1x enrichment, not a demonstrated Brow hotspot.")
FILLING = ("Twenty relative-order cards prescribe three A doubletons and five B doubletons. HX weight eight and HZ weight twelve give 16416 six-round CNOTs versus the parent's 18900. Screen connected verified component incidence graphs against parallel edges and triangles, reserve distinct full-Tanner rooted-refinement fingerprints across slots, then select by worse-basis gate-correlated min-sum failures. Extra slots extend the unused structural catalogue. The surrogate includes independent and shared-opposite-check pair faults but not ASC ordering or noisy syndrome histories. It targets the parent's roughly 35-44 percent failed-shot rates and collective logical flips, without treating distance or decoder settlement as circuit fitness.")
FAMILY_NAME = "r050_dic9_13x15_binary_exposure_decode_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic9 requires q=36")
    # x^9=y^4=1, y*x*y^-1=x^-1.
    # a=x*y^2 and b=y give a^18=1, b^2=a^9, b*a*b^-1=a^-1.
    m, r = 9, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        sign = -1 if bi % 2 else 1
        return ((-sign * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


def _short_logical_counts(checks, stabilizers, n):
    """Exact counts of nontrivial logical supports of weights 1..4.
    Matrices come exclusively from the verified scaffold lift."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(v):
        while v:
            p = v.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            v ^= old
        return False

    columns = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= syndrome_bit
            row ^= bit

    singles = {}
    counts = [0, 0, 0, 0]
    bits = [1 << a for a in range(n)]
    for a, syndrome in enumerate(columns):
        singles.setdefault(syndrome, []).append(a)
        if syndrome == 0 and nontrivial(bits[a]):
            counts[0] += 1

    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = columns[a] ^ columns[b]
            support = bits[a] | bits[b]
            if syndrome == 0 and nontrivial(support):
                counts[1] += 1

            # a < b < c counts each weight-three support once.
            for c in singles.get(syndrome, ()):
                if c > b and nontrivial(support | bits[c]):
                    counts[2] += 1

            bucket = pairs.get(syndrome)
            if bucket is None:
                pairs[syndrome] = [(a, b, support)]
            else:
                # c < d < a < b is the unique ordered partition
                # used to count each weight-four support once.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        counts[3] += 1
                bucket.append((a, b, support))
    return tuple(counts)


def _dic_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_component(G, grid, side):
    """Obtain the exact component from a verified zero-partner probe."""
    zero = [[[]]]
    if side == "A":
        rows = lifted_matrices(G, grid, zero)[0]
    else:
        rows = lifted_matrices(G, zero, grid)[1]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _dic_brief(rows, q, width):
    """Exact graph-cycle screening for a binary incidence component."""
    columns = _dic_columns(rows, width)
    if len(set(columns)) != width:
        return None
    adjacency = [0] * q
    for column in columns:
        if column.bit_count() != 2:
            return None
        bit = column & -column
        a = bit.bit_length() - 1
        b = (column ^ bit).bit_length() - 1
        adjacency[a] |= 1 << b
        adjacency[b] |= 1 << a
    if gf2_rank(rows) != q - 1:
        return None
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        later = neighbors >> (a + 1)
        while later:
            bit = later & -later
            b = a + 1 + bit.bit_length() - 1
            triangles += (neighbors & adjacency[b]).bit_count()
            later ^= bit
    if triangles:
        return None
    opposite_pairs = sum(
        c * (c - 1) // 2
        for a in range(q) for b in range(a)
        for c in [(adjacency[a] & adjacency[b]).bit_count()])
    # Every quadrangle has two opposite vertex pairs.
    return opposite_pairs // 2


def _dic_fingerprint(hx, hz, n):
    """Isomorphism-invariant, uncolored full-Tanner refinement fingerprint.

    Root at EVERY maximum-degree data vertex, not at a presumed
    transitive lift coordinate. Histograms and root multisets are
    invariant under arbitrary qubit/check permutations. Equal
    fingerprints are discarded conservatively; this is not a complete
    graph-isomorphism algorithm.
    """
    from collections import Counter
    import hashlib

    rows = hx + hz
    adjacency = [[] for _ in range(n + len(rows))]
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            adjacency[c].append(n + r)
            adjacency[n + r].append(c)
            row ^= bit
    initial = [(int(v >= n), len(neighbors))
               for v, neighbors in enumerate(adjacency)]
    palette = {s: t for t, s in enumerate(sorted(set(initial)))}
    base = [palette[s] for s in initial]
    maximum = max(len(adjacency[c]) for c in range(n))
    roots = [c for c in range(n) if len(adjacency[c]) == maximum]
    fingerprints = []
    for root in roots:
        colors = list(base)
        colors[root] = max(base) + 1
        digest = hashlib.sha256()
        for _ in range(8):
            signatures = [
                (colors[v], tuple(sorted(colors[w] for w in neighbors)))
                for v, neighbors in enumerate(adjacency)]
            digest.update(repr(sorted(Counter(signatures).items())).encode())
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            refined = [palette[s] for s in signatures]
            if refined == colors:
                break
            colors = refined
        fingerprints.append(digest.hexdigest())
    return hashlib.sha256(
        repr(sorted(fingerprints)).encode()).hexdigest()


def _dic_stress(checks, stabilizers, n, seed):
    """Gate-correlated ideal-syndrome min-sum surrogate, NOT circuit pL."""
    try:
        import numpy as np
    except ImportError:
        return None

    shots = 96
    noise = np.random.default_rng(seed)
    stress = np.repeat(np.array([1.0, 1.6]), shots // 2)
    degree = np.asarray(
        [s.bit_count() for s in _dic_columns(checks, n)], dtype=float)
    other_degree = np.asarray(
        [s.bit_count() for s in _dic_columns(stabilizers, n)], dtype=float)
    exposure = 6.0 * (1.0 + degree + 0.5 * other_degree)
    independent = 1.0 - (1.0 - 0.002) ** (
        exposure[:, None] * stress[None, :])
    errors = noise.random((n, shots)) < independent
    prior = independent.copy()

    # Ancilla pair faults are sampled from actual opposite-check
    # neighborhoods. Their positions do not assert an ASC CNOT order.
    for row in stabilizers:
        ids = []
        while row:
            bit = row & -row
            ids.append(bit.bit_length() - 1)
            row ^= bit
        if len(ids) < 2:
            continue
        probability = np.minimum(
            0.25, 0.002 * 3.0 * (len(ids) - 1) * stress)
        faults = noise.random(shots) < probability
        a = noise.integers(len(ids), size=shots)
        b = noise.integers(len(ids) - 1, size=shots)
        b += b >= a
        active = np.flatnonzero(faults)
        ids = np.asarray(ids, dtype=np.int64)
        errors[ids[a[active]], active] ^= True
        errors[ids[b[active]], active] ^= True
        prior[ids, :] += 2.0 * probability[None, :] / len(ids)
    prior = np.clip(prior, 0.001, 0.35)

    width = max(row.bit_count() for row in checks)
    incidence = np.full((len(checks), width), n, dtype=np.int64)
    for r, row in enumerate(checks):
        t = 0
        while row:
            bit = row & -row
            incidence[r, t] = bit.bit_length() - 1
            row ^= bit
            t += 1
    valid = incidence != n
    padded = np.vstack((errors, np.zeros((1, shots), dtype=bool)))
    syndrome = np.logical_xor.reduce(padded[incidence], axis=1)
    channel = np.vstack((
        np.log((1.0 - prior) / prior),
        np.full((1, shots), 1.0e6)))
    messages = channel[incidence].copy()
    decoded = np.zeros((n, shots), dtype=bool)
    done = np.zeros(shots, dtype=bool)
    positions = np.arange(width)[None, :, None]

    for _ in range(40):
        signs = np.where(messages < 0.0, -1.0, 1.0)
        signs = np.where(valid[:, :, None], signs, 1.0)
        magnitude = np.where(
            valid[:, :, None], np.abs(messages), np.inf)
        smallest = np.partition(magnitude, 1, axis=1)[:, :2, :]
        location = np.argmin(magnitude, axis=1)
        excluded = np.where(
            positions == location[:, None, :],
            smallest[:, 1:2, :], smallest[:, 0:1, :])
        outgoing = (
            0.8 * excluded * signs *
            (np.prod(signs, axis=1) *
             np.where(syndrome, -1.0, 1.0))[:, None, :])
        outgoing = np.where(valid[:, :, None], outgoing, 0.0)
        belief = channel.copy()
        np.add.at(belief, incidence.ravel(),
                  outgoing.reshape((-1, shots)))
        hard = belief < 0.0
        hard[n, :] = False
        reproduced = np.logical_xor.reduce(hard[incidence], axis=1)
        settled = np.all(reproduced == syndrome, axis=0)
        fresh = settled & ~done
        decoded[:, fresh] = hard[:n, fresh]
        done |= settled
        if np.all(done):
            break
        messages = np.clip(
            0.25 * messages +
            0.75 * (belief[incidence] - outgoing), -40.0, 40.0)
        messages = np.where(valid[:, :, None], messages, 1.0e6)

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    failures = int(np.count_nonzero(~done))
    residual = errors ^ decoded
    for shot in np.flatnonzero(done):
        word = sum(1 << int(c)
                   for c in np.flatnonzero(residual[:, shot]))
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                failures += 1
                break
            word ^= pivots[p]
    return failures


def NEW_CTX(rng, G, idx):
    """Reserve new full-Tanner structures, then stress-screen this slot.

    Prefix reconstruction is deterministic and local: no mutable cache
    and no dependence on the order in which the harness builds slots.
    All alternatives reserved for earlier slots remain unavailable.
    """
    q = G.q
    sa, sb = [[2, 2, 2]], [[2, 2, 2, 2, 2]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert sum(sa[0]) + max(sb[0]) == 8
    assert sum(sb[0]) + max(sa[0]) == 12

    a_cards = (
        (18, 18, 4), (18, 9, 4), (18, 4, 4),
        (9, 4, 4), (4, 4, 4))
    b_cards = (
        (18, 9, 4, 4, 4), (18, 18, 4, 4, 4),
        (18, 4, 4, 4, 4), (9, 4, 4, 4, 4))

    def element_order(g):
        word = G.e
        for power in range(1, q + 1):
            word = G.mul(word, g)
            if word == G.e:
                return power
        raise ValueError("broken finite-group element")

    pools = {order: [] for order in (4, 9, 18)}
    for g in G.elements:
        order = element_order(g)
        if order in pools and G.idx(g) < G.idx(G.inv(g)):
            pools[order].append(g)

    def draw(local_rng, card):
        selected = []
        for order in (18, 9, 4):
            selected.extend(
                local_rng.sample(pools[order], card.count(order)))
        local_rng.shuffle(selected)
        return [[[G.e, g] for g in selected]]

    reserved = set()
    finalists = None
    for slot in range(idx + 1):
        card = slot % 20
        ac, bc = a_cards[card // 4], b_cards[card % 4]
        batch = slot // 20
        local_rng = random.Random(360019 + 104729 * slot)
        candidates = {}
        alternatives = []
        checked = set()

        # First obtain diverse, triangle-free connected component pairs.
        # Later rounds widen the search instead of recycling saved labels.
        for wave in range(8):
            for trial in range(48):
                A, B = draw(local_rng, ac), draw(local_rng, bc)
                identity = tuple(
                    tuple(sorted(G.idx(cell[1]) for cell in grid[0]))
                    for grid in (A, B))
                if identity in candidates:
                    continue
                ar = _dic_component(G, A, "A")
                br = _dic_component(G, B, "B")
                aa = _dic_brief(ar, q, 3 * q)
                bb = _dic_brief(br, q, 5 * q)
                if aa is None or bb is None:
                    continue
                # Quadrangle density and total multiplicity are distinct
                # design pressures for additional harness batches.
                if batch % 2:
                    coarse = (aa + bb, max(5 * aa, 3 * bb))
                else:
                    coarse = (max(5 * aa, 3 * bb), aa + bb)
                candidates[identity] = (coarse, A, B)

            ordered = sorted(
                candidates.items(), key=lambda item: (item[1][0], item[0]))
            for identity, (coarse, A, B) in ordered:
                if identity in checked:
                    continue
                checked.add(identity)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 576 or k != 296:
                    continue
                fingerprint = _dic_fingerprint(hx, hz, n)
                if fingerprint in reserved:
                    continue
                if any(fingerprint == entry[1] for entry in alternatives):
                    continue
                alternatives.append(
                    (coarse, fingerprint, A, B, hx, hz, n, k))
                if len(alternatives) == 3:
                    break
            if len(alternatives) == 3:
                break

        if not alternatives:
            raise ValueError("unused triangle-free Tanner catalogue exhausted")
        # Reserve even unselected finalists, so subsequent slots cannot
        # become isomorphic to this slot after its decoder screening.
        reserved.update(entry[1] for entry in alternatives)
        if slot == idx:
            finalists = alternatives

    winner = None
    seed = 1000003 + 1009 * idx
    for coarse, fingerprint, A, B, hx, hz, n, k in finalists:
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        fx = _dic_stress(hz, hx, n, seed)
        fz = _dic_stress(hx, hz, n, seed)
        if fx is None or fz is None:
            # Dependency-free fallback uses exact quantum multiplicity.
            x = _short_logical_counts(hz, hx, n)
            z = _short_logical_counts(hx, hz, n)
            decoding = (max(x[3], z[3]), x[3] + z[3])
        else:
            decoding = (max(fx, fz), fx + fz)
        key = (decoding, coarse, fingerprint)
        if winner is None or key < winner[0]:
            winner = (key, A, B, k, fx, fz)

    key, A, B, k, fx, fz = winner
    assert 19 * 36 == 684 or 576 == (3 * 5 + 1) * q
    assert 19 * 576 <= 40 * k < 21 * 576
    return {"idx": idx, "case": (idx % 20, idx // 20),
            "k": k, "screen": key, "stress_failures": (fx, fz),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Excluding weight-three logicals makes one-sided four-term reinforcement sufficient for better memory: C35 slot 12 still failed 203/250 X shots with d2s=4 and s=3.065, versus all-doubleton slot 0's 154/250 and s=3.275; labels also changed, so this does not isolate term count.
# REFUTED: Difference-order cards alone prevent wasted isomorphic slots: the balanced C35 program retained only 15 distinct codes from 60 submissions, including slot 4 isomorphic to slot 2.
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
