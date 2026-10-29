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
CONCEPT = ("Dic19, order 76, A=(1,2), B=(1,3): quotient-diverse noncommuting trinomials selected using independently rescreened component witnesses and stabilizer dressing should beat the rate-0.300 frontier fitness 30.8.")
GROUP = ("Jump from the order-58 dihedral group to Dic19: x^38=1, y^2=x^19, y*x*y^-1=x^-1. The related order-84 bed reached fitness 30.8 on these shapes. This untried order-76 bed tests a prime-order normal C19 subgroup and C4 quotient; superiority remains a measured-distance hypothesis.")
SHAPES = ("Target cell 0.300 with (1,2)x(1,3): n=532, guaranteed k>=152 and rate>=2/7. Component lengths are 152 and 228 with 76 checks each. Parent measured weight-eight minima were entirely in sector-1 Acol/Bcol blocks. Screen both components and their actual logical embeddings; retain only verified rates in [0.275,0.325).")
FILLING = ("All five cells are distinct-element noncommuting trinomials, giving HX weight 9 and HZ weight 12. Twenty quotient-occupancy policies avoid central-involution-separated terms. Extract components exclusively from verified lifts, exclude component kernels through four, independently rescreen actual kernel witnesses, and lower selection readings using stabilizer-dressed logicals. Exact full-CSS screening through four and cross-slot uncolored incidence fingerprints reject short logicals and conservative orbit duplicates. High witness-search readings are not distance certificates.")
FAMILY_NAME = "r030_dic19_12x13_quotientcards_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 76

def custom_ops(q):
    # Dic19: x^38=1, y^2=x^19, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic19 requires order 76")
    els = [(a, b) for b in range(2) for a in range(38)]

    def red(g):
        a, b = g
        return ((a + 19 * (b // 2)) % 38, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 19*b*d) % 38,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 38, 0) if b == 0 else ((a + 19) % 38, 1)

    def idx(g):
        a, b = red(g)
        return 38*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


from collections import Counter
from functools import lru_cache


def _d19_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _d19_basis(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row
                break
            row ^= pivots[p]
    return pivots


def _d19_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _d19_free4(cols):
    """Exact component exclusion through four, including odd supports."""
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    singles, pairs = set(cols), set()
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles or s in pairs:
                return False
            pairs.add(s)
    return True


def _d19_css_free4(hx, hz, n):
    """Exact logical exclusion through four, modulo actual stabilizers."""
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        basis = _d19_basis(stabilizers)
        cols = _d19_columns(checks, n)
        representatives = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = representatives.get(syndrome)
            if old is not None:
                if _d19_outside(word ^ old, basis):
                    return False
            else:
                representatives[syndrome] = word
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = representatives.get(syndrome)
                if old is not None:
                    if _d19_outside(word ^ old, basis):
                        return False
                else:
                    representatives[syndrome] = word
    return True


def _d19_components(hx, hz, q):
    """Extract both actions from the verified scaffold construction."""
    mask = (1 << q) - 1
    ar = [(row & mask) | (((row >> (3*q)) & mask) << q)
          for row in hx[:q]]
    br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
    return ar, br, _d19_columns(ar, 2*q), _d19_columns(br, 3*q)


def _d19_cycles(rows):
    pairs = {}
    result = 0
    for row in rows:
        ids = []
        while row:
            bit = row & -row
            ids.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(ids):
            for b in ids[:i]:
                key = (b, a)
                result += pairs.get(key, 0)
                pairs[key] = pairs.get(key, 0) + 1
    return result


def _d19_probe(cols, rng, trials, seeds=()):
    """Actual nonzero kernel witnesses; readings are only upper estimates."""
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
    best = min((w.bit_count() for w in memory), default=len(cols)+1)
    order = list(range(len(cols)))
    for trial in range(trials):
        rng.shuffle(order)
        pivots, words = {}, list(memory)
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                hit = pivots.get(p)
                if hit is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= hit[0]
                word ^= hit[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        words = sorted(set(words), key=lambda w: (w.bit_count(), w))
        for i in range(len(words)):
            word = words[i]
            for reducer in words[:i]:
                candidate = word ^ reducer
                if candidate and candidate.bit_count() < word.bit_count():
                    word = candidate
            words[i] = word
            best = min(best, word.bit_count())
        short = sorted(set(words + memory),
                       key=lambda w: (w.bit_count(), w))[:32]
        discoveries = []
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    best = min(best, candidate.bit_count())
                    if candidate.bit_count() <= best + 4:
                        discoveries.append(candidate)
        memory = sorted(set(words + memory + discoveries),
                        key=lambda w: (w.bit_count(), w))[:24]
        if best <= 4:
            break
    return best, memory


def _d19_dressed(item, q):
    """Only verified nontrivial logicals can lower the selection reading."""
    hx, hz = item["hx"], item["hz"]
    mask = (1 << q) - 1
    best = item["n"] + 1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        basis = _d19_basis(stabilizers)
        for seed in sorted(set(seeds), key=int.bit_count)[:5]:
            for fixed in range(3 if side == "A" else 2):
                if side == "A":
                    word = sum(((seed >> (j*q)) & mask)
                               << ((3*j + fixed)*q) for j in range(2))
                else:
                    word = seed << (fixed*3*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _d19_outside(word, basis):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda z: z[0])
                    chosen = word
                    for _, row in nearby[:12]:
                        candidate = word ^ row
                        if candidate.bit_count() < chosen.bit_count():
                            chosen = candidate
                    rows = [row for _, row in nearby[:12]]
                    for i, row in enumerate(rows):
                        for other in rows[:i]:
                            candidate = word ^ row ^ other
                            if candidate.bit_count() < chosen.bit_count():
                                chosen = candidate
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, word.bit_count())
    return best


def _d19_signature(hx, hz, n, k):
    """An uncolored incidence invariant, not an isomorphism algorithm.

    Different signatures certify nonisomorphism. Equal signatures are
    conservatively discarded, even when the codes might be different.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    profiles = [Counter() for _ in rows]
    adjacency = [0] * len(rows)
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] += 1
            if overlap:
                profiles[i][(weights[j], overlap)] += 1
                profiles[j][(weights[i], overlap)] += 1
            if overlap >= 2:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    common = Counter()
    degrees = [a.bit_count() for a in adjacency]
    for i, neighbors in enumerate(adjacency):
        for j in range(i):
            key = (min(degrees[i], degrees[j]),
                   max(degrees[i], degrees[j]),
                   (neighbors & adjacency[j]).bit_count())
            common[key] += 1
    local = Counter((weights[i], tuple(sorted(profile.items())))
                    for i, profile in enumerate(profiles))
    cols = _d19_columns(rows, n)
    return (n, k, tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(overlaps.items())), tuple(sorted(local.items())),
            tuple(sorted(common.items())))


@lru_cache(maxsize=1)
def _d19_bank(G):
    """Deterministic policy bank; caching does not change emitted designs."""
    q = G.q
    assert q == 76 and A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12 and _no_dead_lines(sa, sb)
    policies = ((0,0,1), (0,1,1), (0,1,2), (0,1,3), (0,0,3))
    pools = [[] for _ in range(4)]
    for g in G.elements:
        pools[(2*g[0] + g[1]) % 4].append(g)
    involution = (19, 0)
    fingerprints, bank = set(), []

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", item["n"]+1))
        return (item["k"]*d*d, d, item["da"]+item["db"], -item["cycles"])

    for slot in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + 76019003 + slot)
        card = slot % 20
        a_policy, b_policy = divmod(card, 5)
        finalists, seen = [], set()
        chosen = None

        def draw(side, retry):
            length = 2 if side == "A" else 3
            policy = a_policy if side == "A" else b_policy
            row = []
            for j in range(length):
                recipe = policies[(policy + j + retry + slot//20) % 5]
                selected = None
                for attempt in range(96):
                    phase = rng.randrange(4)
                    direction = rng.choice((-1, 1))
                    cell = []
                    for offset in recipe:
                        available = [
                            g for g in pools[(phase + direction*offset) % 4]
                            if g not in cell and all(
                                G.mul(G.inv(h), g) != involution
                                for h in cell)]
                        if not available:
                            break
                        cell.append(rng.choice(available))
                    if len(cell) != 3:
                        continue
                    relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                    if G.mul(*relatives) == G.mul(*reversed(relatives)):
                        continue
                    selected = cell
                    break
                if selected is None:
                    raise ValueError("noncommuting quotient recipe failed")
                row.append(selected)
            return [row]

        for retry in range(3):
            for attempt in range(64 if retry == 0 else 48):
                A, B = draw("A", retry), draw("B", retry)
                label = tuple(tuple(sorted(cell))
                              for grid in (A, B) for row in grid for cell in row)
                if label in seen:
                    continue
                seen.add(label)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (275*n <= 1000*k < 325*n):
                    continue
                ar, br, ac, bc = _d19_components(hx, hz, q)
                if not _d19_free4(ac) or not _d19_free4(bc):
                    continue
                da, aw = _d19_probe(ac, rng, 4)
                db, bw = _d19_probe(bc, rng, 4)
                item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                            ac=ac, bc=bc, da=da, db=db, aw=aw, bw=bw,
                            cycles=_d19_cycles(ar)+_d19_cycles(br))
                finalists.append(item)
                finalists.sort(key=merit, reverse=True)
                del finalists[6:]

            for item in finalists:
                da, aw = _d19_probe(item["ac"], rng, 20, item["aw"])
                db, bw = _d19_probe(item["bc"], rng, 20, item["bw"])
                item["da"], item["db"] = min(item["da"], da), min(item["db"], db)
                item["aw"], item["bw"] = aw, bw
            finalists.sort(key=merit, reverse=True)
            for item in finalists[:4]:
                da, aw = _d19_probe(item["ac"], rng, 40, item["aw"])
                db, bw = _d19_probe(item["bc"], rng, 40, item["bw"])
                item["da"], item["db"] = min(item["da"], da), min(item["db"], db)
                item["aw"], item["bw"] = aw, bw
                item["logical"] = _d19_dressed(item, q)
            finalists.sort(key=merit, reverse=True)
            for item in finalists:
                hx, hz, n, k = item["hx"], item["hz"], item["n"], item["k"]
                if not _d19_css_free4(hx, hz, n):
                    continue
                signature = _d19_signature(hx, hz, n, k)
                if signature in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= 12
                fingerprints.add(signature)
                chosen = {"cells": {"A": item["A"], "B": item["B"]},
                          "selection": merit(item), "k": k}
                break
            if chosen is not None:
                break
        # Never fill a failed policy with an orbit copy of an earlier code.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this policy's independently optimized, invariant-fresh code."""
    item = _d19_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh short-screened Dic19 filling for policy")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without additional sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled counts with uniform labels reliably suppress short logicals on this bed; many seed slots returned d2s=2 and all five measured seed codes returned d30s=4, with sector-1 minima.
# REFUTED: The shown all-mixed trinomial gap screens reliably exceed measured distance eight; all three shown best programs scored 25.6 with five d30s=8 readings, motivating a different pure/mixed architecture.
# REFUTED: Replacing one mixed trinomial per side with an asymmetric rotation-only triple escaped the measured weight-eight plateau; all five current-parent measurements still returned k=232 and d30s=8.
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
