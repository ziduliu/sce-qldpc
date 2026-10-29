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
CONCEPT = ("C7 semidirect C6 with action 3, order 42, A=(1,3), B=(1,4): quotient-balanced A trinomials and order-six B binomials should produce five measured distance-six codes in cell 0.500, improving its current fitness 12.1.")
GROUP = ("Jump from D8 to the genuinely different Frobenius group C7:C6. Its order-42 related bed reached 23.6 on (1,2)x(3,5), but this high-rate bed is untried. The action 3 has order six modulo seven; order-six relatives avoid the single-binomial weight-two/three/four kernels caused by shorter-order relatives.")
SHAPES = ("Target cell 0.500 with (1,3)x(1,4): n=546, kb=252 and guaranteed rate 6/13. Components have 42 checks and lengths 126/168. A's C6 quotient triples occupy 0,2,4 and force a quotient rank deficit; actual CSS ranks, not a predicted surplus formula, enforce 0.475<=k/n<0.525. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly, with no sector-two minima reported.")
FILLING = ("Twenty cards cross five A relative-exponent patterns with four B trinomial architectures. A counts are (3,3,3); B counts are (2,2,2,3), so both LP row maxima are twelve. Optimize verified component kernels, independently deepen witnesses, test component weight-five dependencies, census complete-matrix quartets, and lower selection readings using verified stabilizer-dressed logicals. Different complete-matrix invariant fingerprints distinguish outputs. Search readings remain upper heuristics, not distance certificates.")
FAMILY_NAME = "r050_C7C6s3_13x14_quotient_rankdef_six_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # x^7=y^6=1, y*x*y^-1=x^3; 3 has order six modulo seven.
    if not valid_q(q):
        raise ValueError("C7:C6 with action 3 requires order 42")
    els = [(a, b) for b in range(6) for a in range(7)]
    powers = [pow(3, b, 7) for b in range(6)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


from functools import lru_cache

# Quotient occupancy is fixed, but relative rotation patterns differ.
_A_PATTERNS = ((1,1,1), (1,1,2), (1,1,3), (1,2,3), (1,2,4))
_B_TRIPLES = ((0,2,4), (0,1,2), (0,1,5), (0,0,1))


def _qcolumns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _qcomponents(hx, hz, q):
    """Extract both actions only from the scaffold's verified lift."""
    mask = (1 << q)-1
    ar = [
        sum(((row >> (j*4*q)) & mask) << (j*q)
            for j in range(3))
        for row in hx[:q]
    ]
    br = [row & ((1 << (4*q))-1) for row in hz[:q]]
    return ar, br, _qcolumns(ar, 3*q), _qcolumns(br, 4*q)


def _qfloor(cols):
    """Exact kernel exclusion through four; five means not excluded."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _qfive(cols):
    """Exact weight-five test after kernel weights <=4 were excluded."""
    pairs = {cols[i] ^ cols[j]
             for i in range(len(cols)) for j in range(i)}
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                if pair ^ cols[h] in pairs:
                    return False
    return True


def _qcycles(rows):
    seen = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for i, a in enumerate(bits):
            for b in bits[:i]:
                key = (b, a)
                old = seen.get(key, 0)
                cycles += old
                seen[key] = old+1
    return cycles


def _qprobe(cols, rng, trials, memory=()):
    """Actual kernel witnesses, recombined across information sets."""
    memory = sorted(set(memory), key=int.bit_count)[:24]
    order = list(range(len(cols)))
    for trial in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = list(memory)
        for c in order:
            syndrome, word = cols[c], 1 << c
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

        words = sorted(set(words), key=int.bit_count)
        for sweep in range(2):
            changed = False
            for i, original in enumerate(words):
                word, weight = original, original.bit_count()
                for reducer in words[:i]:
                    combined = word ^ reducer
                    cw = combined.bit_count()
                    if combined and cw < weight:
                        word, weight = combined, cw
                        changed = True
                words[i] = word
            words.sort(key=int.bit_count)
            if not changed:
                break

        short = sorted(set(words+memory), key=int.bit_count)[:32]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    discoveries.append(combined)
        memory = sorted(set(discoveries), key=int.bit_count)[:24]
        if memory and memory[0].bit_count() <= 4:
            break
    return tuple(memory)


def _qpair_profile(rows, n):
    """Complete-matrix kernel census and permutation-invariant signature.

    Small kernel words are conservatively rejected even if stabilizers.
    With weights <=3 absent, quartet supports give three pair collisions.
    These are kernel counts, not a logical-distance certification.
    """
    cols = _qcolumns(rows, n)
    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old+1
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0)+1
    return (collisions//3, tuple(sorted(histogram.items())),
            tuple(sorted(row.bit_count() for row in rows)))


def _qbasis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _qoutside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _qdressed(item, q):
    """Verify logical embeddings and search lighter stabilizer dressings."""
    hx, hz, n = item["hx"], item["hz"], item["n"]
    mask = (1 << q)-1
    best = n+1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        basis = _qbasis(stabilizers)
        for seed in seeds[:6]:
            for fixed in range(4 if side == "A" else 3):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask) << ((j*4+fixed)*q)
                        for j in range(3))
                else:
                    word = seed << (fixed*4*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _qoutside(word, basis):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        cw = (word ^ row).bit_count()
                        nearby.append((cw, row))
                        if 0 < cw < weight:
                            chosen, weight = word ^ row, cw
                    nearby.sort(key=lambda entry: entry[0])
                    near = [row for _, row in nearby[:12]]
                    for i, row in enumerate(near):
                        for other in near[:i]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _qdraw(rng, G, card):
    """Five A relative patterns crossed with four B term architectures."""
    pattern = _A_PATTERNS[card//4]
    scale = rng.randrange(1, 7)
    A = [[]]
    for delta in pattern:
        u = rng.randrange(7)
        v = (3*u + scale*delta) % 7
        # v != 3*u excludes the commuting C3 subgroup-sum triple.
        A[0].append([G.e, (u, 2), (v, 4)])

    # Three distinct inverse-pair representatives of elements of order six.
    B = [[[G.e, (a, 1)] for a in rng.sample(range(7), 3)]]
    quota = _B_TRIPLES[card % 4]
    for attempt in range(96):
        cell = [G.e, (rng.randrange(7), quota[1]),
                (rng.randrange(7), quota[2])]
        if len(set(cell)) != 3:
            continue
        # Avoid every relative involution; term distinctness alone is weaker.
        if any(
            G.mul(G.mul(G.inv(g), h), G.mul(G.inv(g), h)) == G.e
            for i, g in enumerate(cell) for h in cell[:i]
        ):
            continue
        B[0].append(cell)
        return A, B
    raise ValueError("B trinomial relative-order recipe exhausted")


def _qmerit(item):
    da = min((word.bit_count() for word in item["aw"]),
             default=len(item["ac"])+1)
    db = min((word.bit_count() for word in item["bw"]),
             default=len(item["bc"])+1)
    reading = min(da, db, item.get("logical", item["n"]+1))
    # Actual witnesses are required, so a no-witness sentinel earns nothing.
    if da > len(item["ac"]) or db > len(item["bc"]):
        reading = 0
    return (item["floor"], item.get("clean5", 0),
            item["k"]*reading*reading/item["n"],
            reading, da+db, -item["cycles"])


@lru_cache(maxsize=1)
def _qbank(G):
    """Optimize each structural card and reject equal incidence invariants."""
    assert G.q == 42
    sa, sb = [[3,3,3]], [[2,2,2,3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    bank, used = [], set()

    for slot in range(N_SLOTS):
        card = slot % 20
        local = random.Random(BASE_SEED*31013 + 42131400 + slot)
        pool = []
        exact_seen = set()
        for attempt in range(64 + 8*(slot//20)):
            A, B = _qdraw(local, G, card)
            exact = tuple(tuple(sorted(cell))
                          for grid in (A, B) for row in grid for cell in row)
            if exact in exact_seen:
                continue
            exact_seen.add(exact)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (475*n <= 1000*k < 525*n):
                continue

            ar, br, ac, bc = _qcomponents(hx, hz, G.q)
            fa, fb = _qfloor(ac), _qfloor(bc)
            floor = min(fa, fb)
            if floor < 4:
                continue
            if len(pool) == 6 and floor < pool[-1]["floor"]:
                continue
            aw = _qprobe(ac, local, 6)
            bw = _qprobe(bc, local, 8)
            item = {
                "A": A, "B": B, "hx": hx, "hz": hz,
                "n": n, "k": k, "ac": ac, "bc": bc,
                "aw": aw, "bw": bw, "fa": fa, "fb": fb,
                "floor": floor,
                "cycles": _qcycles(ar)+_qcycles(br),
            }
            pool.append(item)
            pool.sort(key=_qmerit, reverse=True)
            del pool[6:]

        for item in pool:
            item["aw"] = _qprobe(item["ac"], local, 28, item["aw"])
            item["bw"] = _qprobe(item["bc"], local, 36, item["bw"])
            clean_a = item["fa"] == 5 and _qfive(item["ac"])
            clean_b = item["fb"] == 5 and _qfive(item["bc"])
            item["clean5"] = int(clean_a)+int(clean_b)
        pool.sort(key=_qmerit, reverse=True)

        winner = None
        for item in pool:
            px = _qpair_profile(item["hx"], item["n"])
            if px is None:
                continue
            pz = _qpair_profile(item["hz"], item["n"])
            if pz is None:
                continue
            fingerprint = (item["k"], tuple(sorted((px, pz))))
            if fingerprint in used:
                continue

            item["logical"] = _qdressed(item, G.q)
            # A complete quartet overrides optimistic component readings.
            if px[0] or pz[0]:
                item["logical"] = min(item["logical"], 4)
                item["floor"] = min(item["floor"], 4)
            score = _qmerit(item) + (-max(px[0], pz[0]), -px[0]-pz[0])
            if winner is None or score > winner[0]:
                winner = (score, item, fingerprint)

        if winner is None:
            bank.append(None)
            continue
        score, item, fingerprint = winner
        assert css_ok(item["hx"], item["hz"])
        assert max(row.bit_count() for row in item["hx"]+item["hz"]) <= 12
        assert all(len(cell) == len(set(cell))
                   for grid in (item["A"], item["B"])
                   for row in grid for cell in row)
        used.add(fingerprint)
        bank.append((
            tuple(tuple(tuple(cell) for cell in row) for row in item["A"]),
            tuple(tuple(tuple(cell) for cell in row) for row in item["B"]),
            item["k"], score,
        ))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's selected fresh design, never an orbit fallback."""
    item = _qbank(G)[idx]
    if item is None:
        raise ValueError("no fresh rank-legal design for this quotient card")
    A, B, k, score = item
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Adding a second check row and screened distinct relative labels on D8 (2,7)x(2,8) breaks the distance-four plateau; all five measured parent leaders still had d30s=4 and thousands of sector-one weight-four witnesses.
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
