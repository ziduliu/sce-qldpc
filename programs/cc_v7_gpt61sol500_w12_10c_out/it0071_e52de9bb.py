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
CONCEPT = ("Frozen Dic6, order 24, shapes (1,3)x(2,7): order-four rotation-pair A trinomials and redistributed B degree-two columns should yield at least one measured distance reading above six while retaining five distinct codes in cell 0.450.")
GROUP = ("Keep Dic6 byte-for-byte under L1. This bed scores 15.7824 versus 6.957 on the related order-24 C3:C8 bed. Dic6's order-four rotations supply a mixed-triple architecture absent from the parent's A recipes; the unique involution remains excluded from within-cell differences.")
SHAPES = ("Keep A=(1,3), B=(2,7), q=24: n=552, guaranteed k>=240 and rate 10/23, targeting cell 0.450. Components have 72 variables/24 checks and 168 variables/48 checks. All five measured parental minima were weight-six sector-1 Acol/Bcol families across both components. Exact verified ranks retain 0.425<=k/n<0.475.")
FILLING = ("Twenty structural cards cross five A gap-order profiles containing order-four rotations with four zero-containing B geometries. B has four degree-three and three degree-two columns, row totals nine, and single-row degree-two cells use order-twelve rotation differences. Exhaustive component syndrome collisions detect every word through weight six; capped six-word multiplicities and refined actual kernel witnesses guide selection. Complete verified weight-four screening, exact ranks and distinct permutation-invariant fingerprints finish selection. All cells have 0..3 distinct terms and both row caps are twelve; component readings do not certify quantum distance.")
FAMILY_NAME = "r045_dic6_13x27_gap4_zeroaware_exact6_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Dic6: x^12=1, y^2=x^6, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic6 requires order 24")
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        a, b = g
        return ((a + 6 * (b // 2)) % 12, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 6*b*d) % 12,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 12, 0) if b == 0 else ((a + 6) % 12, 1)

    def idx(g):
        a, b = red(g)
        return 12*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)

from functools import lru_cache


def _element_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid element order")


# Mixed triples are distinguished by their rotation-pair order.
# None denotes one broad pure-rotation triple, not a third mixed model.
_TRIPLE_MODELS = (
    (12, 12, 12),
    (6, 6, 6),
    (12, 6, 3),
    (None, 12, 12),
    (None, 6, 6),
)


def _support_recipe(dense):
    # Four or three singleton/singleton columns remain, respectively.
    columns = ([(2, 1), (1, 2), (1, 2)] + [(1, 1)] * 4
               if not dense else
               [(2, 1)] * 2 + [(1, 2)] * 2 + [(1, 1)] * 3)
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    return sa, sb


def _sidon_relatives(rng, G, count):
    """Distinct ordered differences exclude four-cycles among mono columns."""
    for _ in range(64):
        selected = [G.e]
        differences = set()
        pool = list(G.elements)
        rng.shuffle(pool)
        for g in pool:
            if g in selected:
                continue
            added = []
            for h in selected:
                added.extend((G.mul(g, G.inv(h)),
                              G.mul(h, G.inv(g))))
            if len(set(added)) != len(added):
                continue
            if any(d in differences for d in added):
                continue
            selected.append(g)
            differences.update(added)
            if len(selected) == count:
                rng.shuffle(selected)
                return selected
    # A fallback is screened on the complete matrices, not silently accepted.
    return rng.sample(G.elements, count)


def _draw_triple(rng, G, pair_order, order_pools):
    if pair_order is None:
        rotations = [g for g in G.elements if g[1] == 0]
        for _ in range(128):
            cell = rng.sample(rotations, 3)
            orders = [
                _element_order(G, G.mul(G.inv(cell[i]), cell[j]))
                for i in range(3) for j in range(i)
            ]
            if 2 not in orders and 12 in orders:
                break
        else:
            cell = [(0, 0), (1, 0), (3, 0)]
    else:
        a = rng.randrange(12)
        delta = rng.choice(order_pools[pair_order])
        cell = [(a, 0), G.mul((a, 0), delta),
                (rng.randrange(12), 1)]
    # Independent cell gauges diversify sampling but are not counted as designs.
    gauge = G.pick(rng)
    return [G.mul(gauge, g) for g in cell]


def _draw_grids(rng, G, sb, triple_model, double_order, order_pools):
    A = [[_draw_triple(rng, G, kind, order_pools)
          for kind in triple_model]]
    B = [[None] * 7 for _ in range(2)]
    mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
    relatives = _sidon_relatives(rng, G, len(mono))
    for j, relative in zip(mono, relatives):
        top = G.pick(rng)
        # In the verified B action, top * bottom^-1 is the edge relative.
        bottom = G.mul(G.inv(relative), top)
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(7):
            if B[i][j] is not None:
                continue
            g = G.pick(rng)
            B[i][j] = ([g] if sb[i][j] == 1 else
                       [g, G.mul(g, rng.choice(order_pools[double_order]))])
    return A, B


def _syndrome_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _component_columns(hx, hz, q):
    """Extract both components from the scaffold's verified actions."""
    mask = (1 << q) - 1
    ar = []
    for r in range(q):
        row = hx[r]             # A row zero, B column zero
        ar.append(sum(((row >> (j*7*q)) & mask) << (j*q)
                      for j in range(3)))
    br = [row & ((1 << (7*q)) - 1) for row in hz[:2*q]]
    return (_syndrome_columns(ar, 3*q),
            _syndrome_columns(br, 7*q))


def _component_cycles(columns):
    overlaps = {}
    cycles = 0
    for syndrome in columns:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(checks):
            for b in checks[:i]:
                key = (b, a)
                old = overlaps.get(key, 0)
                cycles += old
                overlaps[key] = old + 1
    return cycles


def _kernel_reading(rng, columns, trials):
    """Actual component-kernel witnesses; returned minima are upper bounds."""
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
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        short = sorted(words, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 3:
            break
    return best


def _small_kernel_stats(rows, n):
    """Exact commuting-vector screening through weight three.

    With distinct nonzero column syndromes, equal pair syndromes
    involve disjoint pairs. Every weight-four kernel vector has
    exactly three pair partitions, permitting exact counting.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo

    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    pairs = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            syndrome = a ^ cols[j]
            if syndrome in singles:
                return None
            packed = i * n + j
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = packed
            elif isinstance(previous, int):
                pairs[syndrome] = [previous, packed]
            else:
                previous.append(packed)

    histogram = {}
    participation = [0] * n
    collisions = 0
    for syndrome, value in pairs.items():
        count = 1 if isinstance(value, int) else len(value)
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
        if count > 1:
            collisions += count * (count - 1) // 2
            for packed in value:
                i, j = divmod(packed, n)
                participation[i] += count - 1
                participation[j] += count - 1

    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in cols),
            tuple(p // 3 for p in participation))


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles)


@lru_cache(maxsize=1)
def _design_bank(G):
    """Optimize twenty new structural cards using only verified lifts."""
    gap_cards = (
        (4, 4, 4),
        (4, 4, 12),
        (4, 4, 6),
        (4, 12, 12),
        (4, 6, 12),
    )
    support_cards = (
        ((3, 0), (0, 3), (2, 1), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((2, 1), (2, 1), (1, 2), (1, 2),
         (2, 0), (0, 2), (1, 1)),
        ((3, 0), (3, 0), (0, 3), (0, 3),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (0, 3), (2, 1), (1, 2),
         (2, 0), (0, 2), (1, 1)),
    )
    rotation_pools = {
        order: [g for g in G.elements
                if g[1] == 0 and _element_order(G, g) == order]
        for order in (4, 6, 12)
    }
    double_pools = {
        order: [g for g in G.elements if _element_order(G, g) == order]
        for order in (4, 12)
    }
    q = G.q

    def triple(rng, order, gauged=False):
        # Restrict the designated pair to rotations. In Dic6 many
        # nonrotation elements also have order four.
        cell = [G.e, rng.choice(rotation_pools[order]),
                (rng.randrange(12), 1)]
        if gauged:
            base = G.pick(rng)
            cell = [G.mul(base, g) for g in cell]
        return cell

    def draw_b(rng, sb, phase):
        B = [[None] * 7 for _ in range(2)]
        mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
        relatives = _sidon_relatives(rng, G, len(mono))
        for j, relative in zip(mono, relatives):
            top = G.pick(rng)
            bottom = G.mul(G.inv(relative), top)
            B[0][j], B[1][j] = [top], [bottom]
        for i in range(2):
            for j in range(7):
                if B[i][j] is not None:
                    continue
                count = sb[i][j]
                if count == 0:
                    B[i][j] = []
                elif count == 1:
                    B[i][j] = [G.pick(rng)]
                elif count == 2:
                    # A degree-two column confined to one row is a
                    # binomial cycle: use order twelve, not order four.
                    isolated = sb[1-i][j] == 0
                    order = 12 if isolated else (4, 12)[(phase+i+j) % 2]
                    base = G.pick(rng)
                    difference = rng.choice(double_pools[order])
                    B[i][j] = [base, G.mul(base, difference)]
                else:
                    order = (4, 6, 12)[(phase+i+j) % 3]
                    B[i][j] = triple(rng, order, gauged=True)
        return B

    def six_stats(cols, cap=192):
        """Exact short-word detection, with capped weight-six counting.

        A return flag of seven means no word of weight <=6 exists;
        it is not a distance reading. After excluding weights <=4,
        equal triple syndromes correspond to disjoint triples, and
        each weight-six word has ten unordered triple partitions.
        Saturation never skips the remaining weight-five tests.
        """
        width = len(cols)
        if 0 in cols:
            return (1, cap + 1)
        if len(set(cols)) != width:
            return (2, cap + 1)
        singles = set(cols)
        pairs = set()
        for i, a in enumerate(cols):
            for j in range(i):
                syndrome = a ^ cols[j]
                if syndrome in singles:
                    return (3, cap + 1)
                if syndrome in pairs:
                    return (4, cap + 1)
                pairs.add(syndrome)

        # Uniform odd column parity rules out weight-five kernels.
        odd = all(c.bit_count() & 1 for c in cols)
        triples = {}
        hits = 0
        saturated = False
        for i in range(2, width):
            a = cols[i]
            for j in range(1, i):
                ab = a ^ cols[j]
                for h in range(j):
                    syndrome = ab ^ cols[h]
                    if not odd and syndrome in pairs:
                        return (5, cap + 1)
                    if saturated:
                        continue
                    old = triples.get(syndrome, 0)
                    hits += old
                    triples[syndrome] = old + 1
                    if hits >= 10 * cap:
                        if odd:
                            return (6, cap)
                        saturated = True
                        triples.clear()
        if saturated:
            return (6, cap)
        return (6, hits // 10) if hits else (7, 0)

    def component_merit(item):
        return (item["short"][0], -item["short"][1],
                -item["cycles"])

    def keep(pool, item, size):
        pool.append(item)
        pool.sort(key=component_merit, reverse=True)
        del pool[size:]

    def refine(rng, pool):
        for item in pool:
            reading = _kernel_reading(rng, item["cols"], 36)
            if item["short"][0] <= 6:
                reading = min(reading, item["short"][0])
            item["reading"] = reading

    fingerprints = set()
    bank = []
    dummy_b = [[[G.e] for _ in range(7)] for _ in range(2)]

    for slot in range(N_SLOTS):
        card = slot % 20
        orders = gap_cards[card // 4]
        counts = support_cards[card % 4]
        sb = [[column[i] for column in counts] for i in range(2)]
        assert _grid_weights([[3, 3, 3]], sb) <= 12
        assert _no_dead_lines([[3, 3, 3]], sb)
        rng = random.Random(45024637 + 100003 * slot)
        winner = None

        for batch in range(2):
            # First optimize A independently, avoiding B's short cycles
            # masking differences between the A architectures.
            apool = []
            aseen = set()
            for attempt in range(112):
                A = [[triple(rng, order) for order in orders]]
                signature = tuple(tuple(sorted(cell)) for cell in A[0])
                if signature in aseen:
                    continue
                aseen.add(signature)
                hx, hz, n = lifted_matrices(G, A, dummy_b)
                ac, _ = _component_columns(hx, hz, q)
                short = six_stats(ac)
                item = {
                    "grid": A, "cols": ac, "short": short,
                    "cycles": _component_cycles(ac),
                }
                keep(apool, item, 4)

            if not apool:
                continue
            refine(rng, apool)
            reference_a = max(
                apool, key=lambda item: (
                    item["reading"], component_merit(item)))["grid"]

            bpool = []
            for attempt in range(64):
                B = draw_b(rng, sb, card // 4 + attempt)
                hx, hz, n = lifted_matrices(G, reference_a, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425*n <= 1000*k < 475*n):
                    continue
                _, bc = _component_columns(hx, hz, q)
                short = six_stats(bc)
                item = {
                    "grid": B, "cols": bc, "short": short,
                    "cycles": _component_cycles(bc),
                }
                keep(bpool, item, 6)
            if not bpool:
                continue
            refine(rng, bpool)

            # Evaluate actual joint matrices: component witnesses alone
            # neither determine k nor exclude mixed-sector logicals.
            combinations = [(a, b) for a in apool for b in bpool]
            combinations.sort(
                key=lambda pair: (
                    min(pair[0]["reading"], pair[1]["reading"]),
                    pair[0]["reading"] + pair[1]["reading"],
                    -sum(item["short"][1] for item in pair),
                    -sum(item["cycles"] for item in pair)),
                reverse=True)
            for a, b in combinations[:10]:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425*n <= 1000*k < 475*n):
                    continue
                sx = _small_kernel_stats(hx, n)
                if sx is None:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None:
                    continue
                fingerprint = _incidence_fingerprint(hx, hz, sx, sz)
                if fingerprint in fingerprints:
                    continue
                light = sx[0] + sz[0]
                # Cap optimistic randomized readings; exact short-word
                # witnesses have already reduced the relevant reading.
                d = min(a["reading"], b["reading"], 10)
                objective = (
                    light == 0, k * d * d, d,
                    -light,
                    -(a["short"][1] + b["short"][1]),
                    -(a["cycles"] + b["cycles"]), k)
                if winner is None or objective > winner[0]:
                    winner = (objective, A, B, hx, hz, fingerprint)
            if winner is not None and winner[0][0] and winner[0][2] > 6:
                break

        if winner is None:
            bank.append(None)
            continue
        _, A, B, hx, hz, fingerprint = winner
        if not css_ok(hx, hz):
            raise ValueError("verified lift failed CSS check")
        fingerprints.add(fingerprint)
        bank.append((A, B))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct legal designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this card's zero-aware, exact-six-screened filling."""
    bank = _design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("card found no fresh rate-legal screened filling")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotation binomials plus one trinomial on each D34 (1,4) component raise measured distance above four; all five measured parent codes had d30s=4 and sector-1 minimum logicals.
# REFUTED: Sidon singleton-column relatives, long-order A rotation pairs and randomized component screening eliminate Dic6 weight-six sector-1 families; all five measured parent codes had d30s=6, including the k-kb=10 leader.
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
