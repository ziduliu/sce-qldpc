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
CONCEPT = ("SL(2,3) x C2, order 48, A=(1,2), B=(2,5): primitive-C3-quotient-nonvanishing multiplets and order-six B binomials, selected by stabilizer-tested logical probes, should raise the parent's five measured distance readings beyond twelve.")
GROUP = ("Frozen under L1. This exact bed scored 36.6 and its live record is 36.8; all five parent d30s readings were twelve. The normal Q8 x C2 subgroup of order sixteen gives a C3 quotient for a genuinely new label constraint, without changing the group or claiming a distance certificate.")
SHAPES = ("Frozen (1,2)x(2,5) for cell 0.250: n=576, kb=144, guaranteed rate 1/4; the shape allows q<=49 and this group uses q=48. Component dimensions are 48 checks/96 variables and 96 checks/240 variables. Parent minima overwhelmingly occupied sector-one Acol/Bcol blocks across both A columns and several B columns; slots 4 and 5 had X readings thirteen but limiting Z readings twelve. Verified ranks retain [0.225,0.275).")
FILLING = ("Retain degree-distinguished legal support profiles. Multiplets exclude relative involutions, require noncommuting relative labels and nonzero primitive-C3 quotient parity; B binomials have relative order six. Four occupancy recipes vary quotient and C2 balances. Search A first, then B with shortlisted A components. Deep component embeddings and full CSS information-set/descent probes reject stabilizers exactly and inspect both sectors. Every found witness is an upper bound, never a certificate.")
FAMILY_NAME = "r025_SL23xC2_12x25_degree_component_v2_c3_order6_logical"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Determinant-one 2x2 matrices over F3, times an independent C2.
    # Encoding: (a,b,c,d,z), with a*d-b*c = 1 mod 3.
    if not valid_q(q):
        raise ValueError("SL(2,3) x C2 requires order 48")
    els = [(a,b,c,d,z)
           for z in range(2)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3 == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(g[i] % (2 if i == 4 else 3) for i in range(5))

    def mul(g, h):
        a,b,c,d,z = red(g)
        e,f,u,v,w = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3, (z+w) % 2)

    def inv(g):
        a,b,c,d,z = red(g)
        return (d, (-b) % 3, (-c) % 3, a, z)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

from itertools import combinations_with_replacement
from functools import lru_cache


@lru_cache(maxsize=1)
def _support_designs():
    """Distinct combined Tanner degree fingerprints separate designs."""
    columns = tuple((a, b) for a in range(5) for b in range(5)
                    if 2 <= a+b <= 4)
    candidates = []
    for aa in ((4, 4), (3, 4), (3, 3), (3, 5), (4, 5)):
        for cols in combinations_with_replacement(columns, 5):
            swapped = tuple(sorted((b, a) for a, b in cols))
            if cols > swapped:
                continue
            sb = tuple(tuple(c[i] for c in cols) for i in range(2))
            br = tuple(map(sum, sb))
            bc = tuple(sum(c) for c in cols)
            if min(br) < 5:
                continue
            if sum(aa)+max(bc) > 12 or max(br)+max(aa) > 12:
                continue
            p = [sum((v & 1) << j for j, v in enumerate(row))
                 for row in sb]
            if not p[0] or not p[1] or p[0] == p[1]:
                continue
            checks = tuple(sorted(
                [sum(aa)+v for v in bc] +
                [v+a for a in aa for v in br]))
            qubits = tuple(sorted(
                [a+v for a in aa for v in bc] +
                [sum(aa)+v for v in br]))
            fingerprint = (checks, qubits)
            priority = (min(min(aa), min(bc)), min(aa)+min(bc),
                        sum(aa)+sum(br), -abs(aa[0]-aa[1]),
                        -abs(br[0]-br[1]))
            candidates.append((priority, fingerprint, aa, sb))
    candidates.sort(reverse=True)
    seen, designs = set(), []
    for _, fingerprint, aa, sb in candidates:
        if fingerprint not in seen:
            seen.add(fingerprint)
            designs.append(((aa,), sb))
    if len(designs) < 20:
        raise ValueError("fewer than twenty degree-distinguished designs")
    return tuple(designs)


@lru_cache(maxsize=1)
def _label_pools(G):
    """Exclude equal terms and relative involutions, including central ones."""
    compatible = {}
    for h in G.elements:
        hi = G.inv(h)
        compatible[h] = frozenset(
            g for g in G.elements
            if g != h and G.mul(G.mul(hi, g), G.mul(hi, g)) != G.e)
    return compatible


def _label_grid(rng, G, counts, style):
    compatible = _label_pools(G)
    grid = []
    for row in counts:
        out = []
        for weight in row:
            if not weight:
                out.append([])
                continue
            selected = None
            for restart in range(32):
                offset = rng.randrange(2)
                # Every multi-term cell mixes both direct-factor cosets.
                zs = [(offset+t) % 2 for t in range(weight)]
                if style and weight >= 3:
                    zs = [offset] * (weight-1) + [1-offset]
                rng.shuffle(zs)
                cell = []
                for z in zs:
                    choices = [g for g in G.elements if g[4] == z
                               and all(g in compatible[h] for h in cell)]
                    if not choices:
                        break
                    cell.append(rng.choice(choices))
                if len(cell) == weight:
                    selected = cell
                    break
            if selected is None:
                raise ValueError("could not complete involution-free cell")
            out.append(selected)
        grid.append(out)
    return grid


def _columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _component_columns(hx, hz, q):
    """Extract both component actions only from the verified LP matrices."""
    mask = (1 << q)-1
    ar = [(row & mask) | (((row >> (5*q)) & mask) << q)
          for row in hx[:q]]
    bmask = (1 << (5*q))-1
    br = [row & bmask for row in hz[:2*q]]
    return _columns(ar, 2*q), _columns(br, 5*q)


def _cycles(cols):
    counts, total = {}, 0
    for syndrome in cols:
        active = []
        while syndrome:
            bit = syndrome & -syndrome
            active.append(bit.bit_length()-1)
            syndrome ^= bit
        for i, a in enumerate(active):
            for b in active[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old+1
    return total


def _kernel_probe(rng, cols, trials, best):
    """Actual classical kernel witnesses; no distance certification."""
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
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
                dependencies.append(word)
                best = min(best, word.bit_count())
        short = sorted(dependencies, key=int.bit_count)[:40]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 4:
            break
    return best


def _small_logical(checks, stabilizers, n):
    """Exact existence test for quantum logical supports of weight <=4."""
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

    cols = _columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif outside(word ^ old):
            return True
    for i, syndrome in enumerate(cols):
        bit = 1 << i
        for j in range(i):
            s = syndrome ^ cols[j]
            word = bit | (1 << j)
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif outside(word ^ old):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Optimize quotient/order recipes on degree-distinguished supports."""
    designs = _support_designs()
    sa, sb = designs[idx % len(designs)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    q = G.q
    mask = (1 << q)-1
    mode = idx % 4

    # Compute group-theoretic data; screening matrices still come only
    # from lifted_matrices. Here elements of order <=4 form Q8 x C2.
    orders = {}
    for g in G.elements:
        v = G.e
        for order in range(1, 13):
            v = G.mul(v, g)
            if v == G.e:
                orders[g] = order
                break
        if g not in orders:
            raise ValueError("unexpected group element order")
    normal = [g for g in G.elements if orders[g] in (1, 2, 4)]
    if len(normal) != 16:
        raise ValueError("Q8 x C2 quotient construction failed")
    generator = next(g for g in G.elements if orders[g] == 3)
    grades = {}
    power = G.e
    for grade in range(3):
        for h in normal:
            grades[G.mul(power, h)] = grade
        power = G.mul(power, generator)
    if len(grades) != q:
        raise ValueError("C3 cosets did not partition the group")

    pools = {(a, z): [g for g in G.elements
                     if grades[g] == a and g[4] == z]
             for a in range(3) for z in range(2)}
    compatible = _label_pools(G)
    relative_orders = {
        (g, h): orders[G.mul(G.inv(g), h)]
        for g in G.elements for h in G.elements
    }

    def draw(counts, side):
        grid = []
        for i, row in enumerate(counts):
            out = []
            for j, weight in enumerate(row):
                if not weight:
                    out.append([])
                    continue
                if weight == 1:
                    out.append([G.pick(rng)])
                    continue
                selected = None
                for restart in range(128):
                    shift = (idx+j+2*i+(side == "B")) % 3
                    if mode == 3 and weight >= 3:
                        gs = [rng.randrange(3) for _ in range(weight)]
                    else:
                        templates = {
                            2: (0, 1),
                            3: (0, 0, 1),
                            4: ((0, 0, 0, 1) if mode == 1
                                else (0, 0, 1, 2)),
                            5: ((0, 0, 0, 1, 1) if mode & 1
                                else (0, 0, 1, 1, 2)),
                        }
                        gs = [(a+shift) % 3 for a in templates[weight]]
                    parity = [gs.count(a) & 1 for a in range(3)]
                    # At a primitive cubic root, these two quotient
                    # polynomials vanish. Nonvanishing is not a unit test.
                    if parity in ([0, 0, 0], [1, 1, 1]):
                        continue
                    rng.shuffle(gs)
                    offset = rng.randrange(2)
                    skew = ((mode == 1 and side == "A")
                            or (mode == 2 and side == "B"))
                    if weight == 2:
                        zs = [offset, 1-offset]
                    elif mode == 3:
                        zs = [rng.randrange(2) for _ in range(weight)]
                    elif skew:
                        zs = [offset]*(weight-1)+[1-offset]
                    else:
                        zs = [(offset+t) & 1 for t in range(weight)]
                    rng.shuffle(zs)
                    cell = []
                    for a, z in zip(gs, zs):
                        choices = [g for g in pools[a, z]
                                   if all(g in compatible[h] for h in cell)]
                        if not choices:
                            break
                        cell.append(rng.choice(choices))
                    if len(cell) != weight:
                        continue
                    if side == "B" and weight == 2:
                        if relative_orders[cell[0], cell[1]] != 6:
                            continue
                    if weight >= 3:
                        anchor = G.inv(cell[0])
                        rel = [G.mul(anchor, g) for g in cell[1:]]
                        if not any(G.mul(a, b) != G.mul(b, a)
                                   for t, a in enumerate(rel)
                                   for b in rel[:t]):
                            continue
                    selected = cell
                    break
                if selected is None:
                    raise ValueError("quotient/order recipe exhausted")
                out.append(selected)
            grid.append(out)
        return grid

    def basis(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return tuple(sorted(pivots.items(), reverse=True))

    def normal_form(word, stabilizers):
        # Reduce every pivot, including pivots below a nonpivot bit.
        for p, row in stabilizers:
            if (word >> p) & 1:
                word ^= row
        return word

    def probe(cols, trials, stabilizers=None, embed=None):
        """Actual kernel/logical witnesses; sampled multiplicity is a tie-break.

        With a stabilizer basis, classes are exact quotient normal forms.
        Both single and pair descent preserve a nonzero logical class.
        """
        best = len(cols)+1
        found = set()
        order = list(range(len(cols)))

        def observe(word, cls):
            nonlocal best, found
            if not word or not cls:
                return
            weight = word.bit_count()
            if weight < best:
                best, found = weight, {word}
            elif weight == best:
                found.add(word)

        for _ in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
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
                    cls = word
                    if stabilizers is not None:
                        full = embed(word) if embed is not None else word
                        cls = normal_form(full, stabilizers)
                    words.append((word, cls))
                    observe(word, cls)

            words.sort(key=lambda pair: pair[0].bit_count())
            short = words[:40]
            for t, (word, cls) in enumerate(short):
                for other, oc in short[:t]:
                    observe(word ^ other, cls ^ oc)

            pair_moves = [
                (a ^ b, ac ^ bc)
                for t, (a, ac) in enumerate(words[:12])
                for b, bc in words[:t]
            ]
            seeds = [pair for pair in words if pair[1]][:6]
            for word, cls in seeds:
                weight = word.bit_count()
                for step in range(5):
                    chosen = None
                    for other, oc in words:
                        tc = cls ^ oc
                        if not tc:
                            continue
                        trial = word ^ other
                        tw = trial.bit_count()
                        if tw < weight:
                            chosen = (trial, tc, tw)
                            weight = tw
                    if chosen is None:
                        for other, oc in pair_moves:
                            tc = cls ^ oc
                            if not tc:
                                continue
                            trial = word ^ other
                            tw = trial.bit_count()
                            if tw < weight:
                                chosen = (trial, tc, tw)
                                weight = tw
                    if chosen is None:
                        break
                    word, cls, weight = chosen
                    observe(word, cls)
            if best <= 4:
                break
        return best, len(found)

    # Search the repeatedly limiting A component before spending the
    # joint-search budget on B. The dummy B is only a verified-lift slice.
    skeleton = [[[G.e] for _ in range(5)] for _ in range(2)]
    a_pool = []
    for attempt in range(128):
        A = draw(sa, "A")
        hx, hz, _ = lifted_matrices(G, A, skeleton)
        ac, _ = _component_columns(hx, hz, q)
        da, hits = probe(ac, 6)
        cycles = _cycles(ac)
        key = (da, -hits, -cycles)
        a_pool.append(dict(A=A, ac=ac, da=da, hits=hits,
                           cycles=cycles, key=key))
        a_pool.sort(key=lambda item: item["key"], reverse=True)
        del a_pool[4:]

    for item in a_pool:
        da, hits = probe(item["ac"], 64)
        item["da"] = min(item["da"], da)
        item["key"] = (item["da"], -hits, -item["cycles"])
    a_pool.sort(key=lambda item: item["key"], reverse=True)

    finalists = []

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d/item["n"], d,
                item["da"]+item["db"], -item["hits"], -item["cycles"])

    for attempt in range(64):
        ai = a_pool[attempt % len(a_pool)]
        A, B = ai["A"], draw(sb, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k > 0 and 225*n <= 1000*k < 275*n):
            continue
        ac, bc = _component_columns(hx, hz, q)
        db, hits = probe(bc, 6)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ac, bc=bc, da=ai["da"], db=db,
                    hits=hits, cycles=ai["cycles"]+_cycles(bc))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[4:]

    if not finalists:
        raise ValueError("no exact-rate quotient/order candidate")

    embed_a = lambda word: ((word & mask)
                            | ((word >> q) << (5*q)))
    for item in finalists:
        bx, bz = basis(item["hx"]), basis(item["hz"])
        # A-kernel embeddings are Z candidates at Bcol0; B-kernel
        # embeddings are X candidates at Acol0. Reject stabilizers.
        da, ha = probe(item["ac"], 64, bz, embed_a)
        db, hb = probe(item["bc"], 64, bx)
        # Unlike component probes, these inspect both LP sectors.
        dx, hdx = probe(_columns(item["hz"], item["n"]), 12, bx)
        dz, hdz = probe(_columns(item["hx"], item["n"]), 12, bz)
        item["da"], item["db"] = min(da, dz), min(db, dx)
        item["hits"] = ha+hb+hdx+hdz
        item["light"] = (
            _small_logical(item["hx"], item["hz"], item["n"]) or
            _small_logical(item["hz"], item["hx"], item["n"]))

    winner = max(finalists, key=lambda item: (
        not item["light"],) + merit(item))
    assert css_ok(winner["hx"], winner["hz"])
    assert max(row.bit_count()
               for row in winner["hx"]+winner["hz"]) <= W_MAX
    return {"A": winner["A"], "B": winner["B"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete quotient/order-constrained selected cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotational binomials plus a trinomial on D34 (1,4)x(1,4) did not raise the distance reading above four: all 60 distinct slots had d2s=4 and all five measured slots had d30s=4.
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
