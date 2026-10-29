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
CONCEPT = ("C13 semidirect C3 with action 3, order 39, A=(1,2), B=(3,5): degree-distinguished supports, packed A differences and order-thirteen B binomials should produce five measured distance readings >=15 in cell 0.150.")
GROUP = ("Jump from Dic11 to C13:C3, x^13=y^3=1 and y*x*y^-1=x^3. The recorded order-39 centre-one/seven-class (2,3)x(2,3) bed reached fitness 45.2; its transfer to this untried shape bed is a hypothesis. This odd-order group has no involutions: nonidentity rotations have order thirteen and all elements outside C13 have order three.")
SHAPES = ("Keep (1,2)x(3,5), targeting 0.150: n=13*39=507, kb=78 and guaranteed rate 2/13. Component variable/check counts are 78/39 and 195/117. Parent limiting Z witnesses used both A columns across Bcol1-3, with measured minima 13-14 versus larger X readings. Verified ranks require 78<=k<=88, entirely inside the target band.")
FILLING = ("Enumerate twenty supports with different combined-Tanner degree fingerprints, obeying both check-weight sums <=12. Three-coset A models are selected to reduce repeated directed pair differences; every B binomial has an order-thirteen relative element, avoiding an order-three binomial kernel. Verified component extraction, cross-information-set kernel recombination, stabilizer-tested embeddings and full-code logical probes select fresh fillings. Exact logical exclusion through weight four remains mandatory. Different full-Tanner invariants distinguish later slots. All sampled witness readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r015_frob13c3_12x35_pairpack_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # x^13=y^3=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    if not valid_q(q):
        raise ValueError("C13:C3 requires order 39")
    els = [(a, b) for b in range(3) for a in range(13)]
    action = (1, 3, 9)

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 13, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=1)
def _f39_support_plans():
    """Different degree histograms prove nonisomorphism of these cards."""
    from itertools import combinations_with_replacement
    choices = {}
    for ap in ((4, 4), (4, 5), (3, 5), (3, 4), (3, 3)):
        ar = sum(ap)
        for masks in combinations_with_replacement(range(8), 5):
            sb = tuple(tuple(1 + ((m >> i) & 1) for m in masks)
                       for i in range(3))
            br = tuple(sum(row) for row in sb)
            bc = tuple(3 + m.bit_count() for m in masks)
            if ar + max(bc) > W_MAX:
                continue
            if max(ap) + max(br) > W_MAX:
                continue
            checks = tuple(sorted(
                [ar + c for c in bc]
                + [r + a for a in ap for r in br]))
            qubits = tuple(sorted(
                [a + c for a in ap for c in bc]
                + [ar + r for r in br]))
            fingerprint = (checks, qubits)
            priority = (min(ap), min(bc), ar + sum(br),
                        -max(br), -abs(ap[0] - ap[1]))
            entry = (priority, ap, sb)
            old = choices.get(fingerprint)
            if old is None or entry > old:
                choices[fingerprint] = entry
    ordered = sorted(choices.values(), reverse=True)
    if len(ordered) < 20:
        raise ValueError("fewer than twenty degree-distinguished supports")
    return tuple((ap, sb) for _, ap, sb in ordered[:20])


def _d11_support(card):
    """Legacy helper name; counts now belong to the order-39 bed."""
    ap, bp = _f39_support_plans()[card % 20]
    sa = [list(ap)]
    sb = [list(row) for row in bp]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _d11_a_cell(rng, G, weight, mode):
    """Fresh internal labels with prescribed three-coset occupancies."""
    models = (
        (weight - 2, 1, 1),
        ((weight + 1) // 2, weight // 2, 0),
        (1, weight - 1, 0),
        (1, (weight - 1) // 2, weight // 2),
    )
    counts = models[mode % 4]
    phase = rng.randrange(3)
    cell = []
    for b, count in enumerate(counts):
        pool = [(a, (b + phase) % 3) for a in range(13)]
        cell.extend(rng.sample(pool, count))
    assert len(cell) == weight and len(set(cell)) == weight
    return cell


def _d11_draw(rng, G, sa, sb, mode):
    """Pack A pair differences; B binomials never have order three.

    Pair differences are a construction filter, not a replacement lift.
    The actual component cycle counts come from lifted_matrices.
    """
    used = set()
    arow = [None] * len(sa[0])
    order = list(range(len(arow)))
    rng.shuffle(order)
    order.sort(key=lambda j: -sa[0][j])
    for j in order:
        best = None
        for attempt in range(48):
            cell = _d11_a_cell(rng, G, sa[0][j], mode)
            differences = [
                G._key(G.mul(g, G.inv(h)))
                for g in cell for h in cell if g != h
            ]
            unique = set(differences)
            cost = (len(differences) - len(unique)
                    + 2*len(unique & used))
            key = (cost, len(unique & used))
            if best is None or key < best[0]:
                best = (key, cell, unique)
            if cost == 0:
                break
        arow[j] = best[1]
        used.update(best[2])

    rotations = [(a, 0) for a in range(1, 13)]
    B = []
    for counts in sb:
        row = []
        for count in counts:
            g = G.pick(rng)
            cell = [g]
            if count == 2:
                cell.append(G.mul(g, rng.choice(rotations)))
            assert len(cell) == count and len(set(cell)) == count
            row.append(cell)
        B.append(row)
    return [arow], B


def _d11_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _d11_components(hx, hz, q):
    """Extract the actual component actions from the verified LP lift."""
    mask = (1 << q) - 1
    ar = [(row & mask) | (((row >> (5*q)) & mask) << q)
          for row in hx[:q]]
    bm = (1 << (5*q)) - 1
    br = [row & bm for row in hz[:3*q]]
    return (_d11_columns(ar, 2*q), _d11_columns(br, 5*q))


def _d11_floor4(cols):
    """Exact kernel screening through four; five is only a sentinel."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            syndrome = ci ^ cj
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _d11_cycles(cols):
    overlaps = {}
    total = 0
    for col in cols:
        checks = []
        while col:
            low = col & -col
            checks.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(checks):
            for b in checks[:i]:
                pair = (b, a)
                old = overlaps.get(pair, 0)
                total += old
                overlaps[pair] = old + 1
    return total


def _d11_kernel_reading(cols, rng, trials, accept=None, initial=None):
    """Exact kernel witnesses with cross-information-set recombination.

    For full-code probes, accept rejects stabilizers. An initial reading
    must come from an already verified logical witness, not a target.
    """
    best = len(cols) + 1 if initial is None else initial
    witnesses = set()
    memory = []
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        if weight > best + 2:
            return
        if accept is not None and not accept(word):
            return
        if weight < best:
            best = weight
        if weight <= best + 2:
            witnesses.add(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                observe(word)

        words.sort(key=int.bit_count)
        discoveries = set(memory + words[:24])
        cutoff = best + 4
        for i, word in enumerate(words):
            for other in words[:i]:
                combined = word ^ other
                if combined and combined.bit_count() <= cutoff:
                    observe(combined)
                    discoveries.add(combined)

        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined and combined.bit_count() <= best + 4:
                    observe(combined)
                    discoveries.add(combined)

        reducers = words + memory
        for start in words[:6] + memory[:4]:
            word = start
            for _ in range(4):
                improved = word
                weight = word.bit_count()
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
                discoveries.add(word)

        memory = sorted(discoveries,
                        key=lambda v: (v.bit_count(), v))[:32]
        witnesses = set(sorted(
            witnesses, key=lambda v: (v.bit_count(), v))[:64])
        if best <= 4:
            break
    short = sorted(witnesses, key=lambda v: (v.bit_count(), v))[:32]
    return best, short


def _d11_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _d11_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _d11_small_logical(checks, stabilizers, n):
    """Exactly detect a non-stabilizer logical of weight one through four."""
    basis = _d11_basis(stabilizers)
    cols = _d11_columns(checks, n)
    representatives = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        previous = representatives.get(syndrome)
        if previous is None:
            representatives[syndrome] = word
        elif _d11_outside(word ^ previous, basis):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            previous = representatives.get(syndrome)
            if previous is None:
                representatives[syndrome] = word
            elif _d11_outside(word ^ previous, basis):
                return True
    return False


def _d11_logical_reading(words, side, hx, hz, q, n):
    """Validate component embeddings against the full opposite checks."""
    mask = (1 << q) - 1
    checks, stabilizers = (hx, hz) if side == "A" else (hz, hx)
    basis = _d11_basis(stabilizers)
    best = n + 1
    for word in words:
        weight = word.bit_count()
        if weight >= best:
            continue
        for block in range(5 if side == "A" else 2):
            if side == "A":
                full = (((word & mask) << (block*q)) |
                        ((word >> q) << ((5 + block)*q)))
            else:
                full = word << (block*5*q)
            if any((full & row).bit_count() % 2 for row in checks):
                raise ValueError("verified component embedding failed")
            if _d11_outside(full, basis):
                best = weight
                break
    return best


def _d11_fingerprint(hx, hz, n):
    """Combined-Tanner invariant, ignoring even the X/Z check colouring.

    Different fingerprints imply nonisomorphic incidence graphs.
    Matching fingerprints conservatively reject some nonisomorphic codes.
    No label, group encoding, or protograph ordering enters this invariant.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, row in enumerate(rows):
        while row:
            low = row & -row
            incidents[low.bit_length() - 1].append(r)
            row ^= low

    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = {}
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] = square.get(t, 0) + multiplicity*other
        triangles = sum(neighbors.get(t, 0)*value
                        for t, value in square.items())
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            triangles,
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _d11_bank(G):
    """Optimize prescribed cards jointly and retain invariant-distinct codes."""
    bank = []
    fingerprints = set()

    for slot in range(N_SLOTS):
        card = slot % 20
        mode = (card + slot // 20) % 4
        sa, sb = _d11_support(card)
        rng = random.Random(15039013 + 100003*slot)
        chosen = None

        # The second batch explores new internal labels, not orbit images.
        for batch in range(2):
            finalists = []
            for attempt in range(36):
                A, B = _d11_draw(rng, G, sa, sb, mode)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 507 or not 78 <= k <= 88:
                    continue
                ac, bc = _d11_components(hx, hz, G.q)
                fa, fb = _d11_floor4(ac), _d11_floor4(bc)
                if min(fa, fb) < 5:
                    continue
                da, wa = _d11_kernel_reading(ac, rng, 6)
                db, wb = _d11_kernel_reading(bc, rng, 6)
                cycles = _d11_cycles(ac) + _d11_cycles(bc)
                floor = min(da, db, 20)
                key = (k*floor*floor/n, floor, da + db, -cycles)
                finalists.append({
                    "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db, "wa": wa, "wb": wb,
                    "cycles": cycles,
                })
                finalists.sort(key=lambda item: item["key"], reverse=True)
                del finalists[4:]

            deep = []
            for item in finalists:
                da, wa = _d11_kernel_reading(item["ac"], rng, 48)
                db, wb = _d11_kernel_reading(item["bc"], rng, 32)
                item["da"] = min(item["da"], da)
                item["db"] = min(item["db"], db)
                wa = sorted(set(wa + item["wa"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                wb = sorted(set(wb + item["wb"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                dz = _d11_logical_reading(
                    wa, "A", item["hx"], item["hz"], G.q, item["n"])
                dx = _d11_logical_reading(
                    wb, "B", item["hx"], item["hz"], G.q, item["n"])
                if max(dx, dz) > item["n"]:
                    continue
                d = min(dx, dz, item["da"], item["db"], 20)
                key = (item["k"]*d*d/item["n"], d,
                       dx + dz, -item["cycles"])
                deep.append((key, item))

            deep.sort(key=lambda pair: pair[0], reverse=True)
            joint = []
            for key, item in deep[:3]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xs, zs = _d11_basis(hx), _d11_basis(hz)
                # key[1] is no larger than the stabilizer-tested
                # embedded logical readings already found above.
                upper = key[1]
                dx, _ = _d11_kernel_reading(
                    _d11_columns(hz, n), rng, 6,
                    accept=lambda word, basis=xs:
                        _d11_outside(word, basis),
                    initial=upper)
                dz, _ = _d11_kernel_reading(
                    _d11_columns(hx, n), rng, 6,
                    accept=lambda word, basis=zs:
                        _d11_outside(word, basis),
                    initial=upper)
                d = min(upper, dx, dz)
                joint_key = (item["k"]*d*d/n, d,
                             key[2], -item["cycles"])
                joint.append((joint_key, item))

            joint.sort(key=lambda pair: pair[0], reverse=True)
            for key, item in joint:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _d11_small_logical(hx, hz, n):
                    continue
                if _d11_small_logical(hz, hx, n):
                    continue
                fingerprint = _d11_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert 125*n <= 1000*item["k"] < 175*n
                chosen = (item["A"], item["B"], fingerprint)
                break
            if chosen is not None:
                break

        if chosen is None:
            bank.append(None)
        else:
            A, B, fingerprint = chosen
            fingerprints.add(fingerprint)
            bank.append((A, B))

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign one independently optimized, incidence-distinct card."""
    selected = _d11_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rate-legal filling passed logical-four screening")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: D8 (2,7)x(2,8) distinct monomial relatives and short-syndrome selection did not break the distance-four plateau: all five measured readings were 4, with both logical types entirely in sector 1.
# REFUTED: Dic11 (1,2)x(3,5) degree tradeoffs and component selection produce five readings >=14; measured slots 11 and 28 remained at 13, although the other three measured readings were 14.
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
