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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dic7, order 28, shapes (2,4)x(2,4): independently optimized saturated constituents, selected for kernel witnesses and rank deficits, should supply five novel measured codes exceeding fitness 29.9 in rate cell 0.200; larger-distance readings remain upper bounds.")
GROUP = ("Keep the Dic7 presentation and order 28 unchanged. Its recorded (2,4)x(2,4), n=560 bed reached fitness 29.0. Independent constituent optimization improved the related Dic11 bed, motivating stronger selection here without transferring a distance guarantee.")
SHAPES = ("Change to equal (2,4) sides: each constituent has 112 variables and 56 checks, n=560, kb=112 and guaranteed rate 0.200. The size window permits q<=29; unchanged q=28 fits. Actual ranks enforce cell 0.200. Parent weight-three minima occupied sector-1 Acol/Bcol blocks, especially its unit pivots; the new cubic columns remove that explicit three-term cancellation mechanism. This deliberately redirects from the high-rate distance-three plateau to the near-record rate-0.200 bed.")
FILLING = ("Twenty complementary support/order prescriptions have six terms per row and three per column, giving BOTH CSS check maxima nine, also below twelve. Independently optimize constituents using verified lifts, exact kernel rejection through four, randomized kernel witnesses and rank deficits. Pair equally refined finalists, require distinct intrinsic uncoloured Tanner invariants, enforce actual ranks and complete-matrix kernel rejection through four, and verify nontrivial logical witnesses before stabilizer reduction. Rank deficits may increase k above 112; longer witness readings are upper bounds, not distance certificates. Unseen archive novelty remains evaluator-tested.")
FAMILY_NAME = "r020_dic7_24x24_saturated_independent_rankwitness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic7 requires order 28")
    # x^14=1, y^2=x^7, y*x*y^-1=x^-1.
    # (a,b) encodes x^a y^b, with b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (g[0] % 14, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1) * c + 7*b*d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + 7) % 14, b)

    def idx(g):
        a, b = red(g)
        return 14*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


_DIC_PROFILES = (
    (4, 4, 4), (4, 4, 7), (4, 4, 14), (4, 7, 7),
    (4, 7, 14), (4, 14, 14), (7, 7, 7), (7, 7, 14),
    (7, 14, 14), (14, 14, 14),
)
_DIC_PRIMARY = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 4), (0, 9), (1, 5), (1, 8), (2, 3),
    (2, 7), (3, 5), (4, 6), (5, 8), (6, 9),
)
_DIC_CARDS = _DIC_PRIMARY + tuple(
    (a, b) for a in range(10) for b in range(a, 10)
    if (a, b) not in _DIC_PRIMARY
)


def _dic_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid finite-group order")


def _dic_component(rng, G, orders, pools):
    """A fixed monomial pivot plus three prescribed-order binomials."""
    row = [[G.e]]
    used = set()
    for order in orders:
        available = [h for h in pools[order]
                     if h not in used and G.inv(h) not in used]
        if not available:
            raise ValueError("relative-order card exhausted")
        h = rng.choice(available)
        used.add(h)
        g = G.pick(rng)
        row.append([g, G.mul(g, h)])
    return [row]


def _dic_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_excludes_two(rows, n):
    """Complete exclusion of commuting supports of weights one and two."""
    columns = _dic_columns(rows, n)
    return 0 not in columns and len(set(columns)) == n


def _dic_three_witness(G, B):
    """X witness in Acol1: one B-binomial bit and two B-pivot bits.

    B[0][0]=identity. For binomial labels u,v, selecting its column
    at identity has check syndromes at u^-1 and v^-1, canceled by
    those two columns of the monomial pivot.
    """
    q = G.q
    u, v = B[0][1]
    base = 4*q                  # Acol1/Bcol0
    return ((1 << (base + q + G.idx(G.e)))
            | (1 << (base + G.idx(G.inv(u))))
            | (1 << (base + G.idx(G.inv(v)))))


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _component_probe(rows, n, rng):
    """Exact search through weight 4, then heuristic kernel probes."""
    from itertools import combinations

    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2

    pairs = set()
    found4 = False
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    if found4:
        return 4

    best = n + 1
    for _ in range(4):
        order = list(range(n))
        rng.shuffle(order)
        image = [1 << order[c] for c in range(n)]
        pivots = {}
        for original in rows:
            row = 0
            bits = original
            while bits:
                bit = bits & -bits
                row ^= image[bit.bit_length() - 1]
                bits ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for f in range(n):
            if f in pivots:
                continue
            word = 1 << f
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            basis.append(word)

        for word in basis:
            best = min(best, word.bit_count())
        for a, b in combinations(basis, 2):
            best = min(best, (a ^ b).bit_count())
        for a, b, c in combinations(basis, 3):
            best = min(best, (a ^ b ^ c).bit_count())
        if best <= 5:
            break
    return best


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


from functools import lru_cache


def _sat_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _sat_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _sat_clean4(rows, width):
    """Exactly exclude every nonzero kernel support of weight <=4."""
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            value = first ^ second
            if value in singles or value in pairs:
                return False
            pairs.add(value)
    return True


def _sat_probe(rows, width, local, passes, seed=0):
    """Return an actual kernel witness, never a distance certificate."""
    cols = _dic_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1
    order = list(range(width))

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        # Elementary exchanges preserve the complete kernel span.
        for i in range(len(kernel)):
            for j in range(len(kernel)):
                if i != j:
                    word = kernel[i] ^ kernel[j]
                    if 0 < word.bit_count() < kernel[i].bit_count():
                        kernel[i] = word
            inspect(kernel[i])
        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for third in short[:j]:
                    inspect(pair ^ third)
        for _ in range(2):
            previous = best
            for i, first in enumerate(short):
                inspect(previous ^ first)
                for second in short[:i]:
                    inspect(previous ^ first ^ second)
            if previous == best:
                break
        if weight <= 6:
            break
    if not best:
        raise ValueError("constituent kernel probe found no witness")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _sat_signature(rows, width):
    """Intrinsic invariant of the full uncoloured Tanner graph.

    Unequal keys prove inequivalence. Equality is inconclusive and is
    rejected conservatively. No group or protograph labels enter it.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(rows)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * m
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlaps.append((i, j, value))
            if value:
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i, j, value in overlaps:
        common = (
            value,
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(weights[j],) + common] += 1
        profiles[j][(weights[i],) + common] += 1

    roots = []
    for root in range(m):
        visited = frontier = 1 << root
        shells = []
        # Truncated rooted shells are still exact graph invariants.
        for _ in range(3):
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= adjacency[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~visited
            visited |= frontier
            shells.append(frontier.bit_count())
        roots.append((
            weights[root], tuple(sorted(profiles[root].items())),
            tuple(shells),
        ))
    cols = _dic_columns(rows, width)
    signature = (
        width, m,
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(roots)),
    )
    return sha256(repr(signature).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _sat_portfolio(G, count):
    """Immutable deterministic optimization of twenty structural cards.

    Memoization changes build cost only. CELL receives immutable grids,
    and no slot can alter the construction selected for another slot.
    """
    q, e = G.q, G.e
    assert q == 28 and A_SHAPE == B_SHAPE == (2, 4)
    width = 4 * q
    cap = min(W_MAX, 9)
    profiles = (
        (1, 1, 2, 2),
        (0, 1, 2, 3),
        (0, 2, 2, 2),
        (0, 0, 3, 3),
    )
    orders = {g: _dic_order(G, g) for g in G.elements}
    pools = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (2, 4, 7, 14)
    }
    directions = [g for g in G.elements if g != e]
    rotations = pools[7] + pools[14]
    cross = pools[4]

    def component_rows(grid, right=False):
        # A zero auxiliary side isolates the exact scaffold action.
        if right:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        assert len(rows) == 2*q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def draw(local, profile, policy, trial):
        counts = (profile, tuple(3 - c for c in profile))
        grid = []
        for i, row in enumerate(counts):
            output = []
            for j, terms in enumerate(row):
                anchor = G.pick(local)
                if terms == 0:
                    cell = []
                elif terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    requested = (4, 7, 14, 2)[policy % 4]
                    if policy == 4:
                        requested = (4, 7, 14, 2)[
                            (trial + i + j) % 4]
                    # Mixed central/noncentral reinforcement is a
                    # deliberate fifth prescription, not an orbit copy.
                    if policy == 3 and (i + j + trial) % 2:
                        requested = 4
                    relative = local.choice(pools[requested])
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    r = local.choice(rotations)
                    s = local.choice(cross)
                    if policy == 3:
                        r = local.choice(pools[2])
                    if policy == 4 and (trial + j) % 2:
                        r = local.choice(directions)
                        while r == s:
                            r = local.choice(directions)
                    cell = [anchor, G.mul(anchor, r),
                            G.mul(anchor, s)]
                assert len({G._key(g) for g in cell}) == terms
                output.append(cell)
            grid.append(output)
        return grid

    bank, component_keys = [], set()
    for card in range(20):
        profile, policy = divmod(card, 5)
        counts = (profiles[profile],
                  tuple(3-c for c in profiles[profile]))
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) == 9 <= cap
        local = random.Random(BASE_SEED ^ (0xD72400 + 104729*card))
        records, fillings = [], set()

        for trial in range(144):
            grid = draw(local, profiles[profile], policy, trial)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = component_rows(grid)
            rank = gf2_rank(rows)
            if not _sat_clean4(rows, width):
                continue
            cycles = _four_cycles(rows)
            records.append(dict(
                grid=grid, rows=rows, rank=rank, cycles=cycles,
                card=card, trial=trial))

        # Retain several rank strata: rank deficits must not win by
        # receiving weaker distance scrutiny than full-rank candidates.
        strata = {}
        for record in records:
            strata.setdefault(record["rank"], []).append(record)
        shortlist = []
        for rank in sorted(strata):
            bucket = sorted(strata[rank],
                            key=lambda r: (r["cycles"], r["trial"]))
            shortlist.extend(bucket[:6])
        shortlist = sorted(
            shortlist, key=lambda r: (r["cycles"], r["rank"], r["trial"])
        )[:18]
        for record in shortlist:
            record["word"] = _sat_probe(
                record["rows"], width, local, 4)
            record["quality"] = (
                record["word"].bit_count(),
                2*q-record["rank"], -record["cycles"])
        shortlist.sort(key=lambda r: r["quality"], reverse=True)
        finalists = shortlist[:5]
        for record in finalists:
            record["word"] = _sat_probe(
                record["rows"], width, local, 20, record["word"])
            record["quality"] = (
                record["word"].bit_count(),
                2*q-record["rank"], -record["cycles"])
        finalists.sort(key=lambda r: r["quality"], reverse=True)

        kept = 0
        for record in finalists:
            signature = _sat_signature(record["rows"], width)
            if signature in component_keys:
                continue
            right = component_rows(record["grid"], True)
            right_word, source = 0, record["word"]
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length()-1, q)
                right_word |= 1 << (
                    j*q + G.idx(G.inv(G.elements[h])))
                source ^= bit
            # The inversion transfer is verified, not assumed.
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right)
            assert gf2_rank(right) == record["rank"]
            record["right_word"] = right_word
            component_keys.add(signature)
            bank.append(record)
            kept += 1
            if kept == 3:
                break
    if len(bank) < 5:
        raise ValueError("too few saturated kernel-four-clean constituents")

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            weak = min(a["word"].bit_count(), b["word"].bit_count())
            priority = (
                weak, 4*q-a["rank"]-b["rank"],
                -a["cycles"]-b["cycles"],
                int(a["card"] != b["card"]))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def quantum_word(record, kind, checks, stabilizers):
        """Verify a logical before reducing its stabilizer coset."""
        own = _sat_basis(stabilizers)
        best = 0
        for block in range(4):
            if kind == "X":
                word = record["right_word"] << (block*width)
            else:
                word, source = 0, record["word"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*4+block)*q+h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if _sat_outside(word, own):
                if not best or word.bit_count() < best.bit_count():
                    best = word
        if not best:
            return 0
        for _ in range(2):
            previous = best
            weight = best.bit_count()
            for i, first in enumerate(stabilizers):
                trial = previous ^ first
                if trial.bit_count() < weight:
                    best, weight = trial, trial.bit_count()
                for second in stabilizers[:i]:
                    trial = previous ^ first ^ second
                    if trial.bit_count() < weight:
                        best, weight = trial, trial.bit_count()
            if best == previous:
                break
        assert best and _sat_outside(best, own)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
        return best

    results, graph_keys = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 560
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (175*n <= 1000*k < 225*n):
            continue
        signature = _sat_signature(hx+hz, n)
        if signature in graph_keys:
            continue
        tested += 1
        if not (_sat_clean4(hx, n) and _sat_clean4(hz, n)):
            continue
        assert max(row.bit_count() for row in hx+hz) <= cap
        if not css_ok(hx, hz):
            raise ValueError("verified saturated lift failed CSS")
        xword = quantum_word(b, "X", hz, hx)
        zword = quantum_word(a, "Z", hx, hz)
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())
        if reading <= 4:
            raise ValueError("witness contradicted exact full-kernel screen")

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        graph_keys.add(signature)
        quality = (k*reading*reading/n, reading, k, preliminary)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (xword, zword)))
        if len(results) >= max(60, 3*count):
            break
        if tested >= max(320, 8*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of the immutable screened portfolio."""
    catalogue = _sat_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct saturated Dic7 portfolio exhausted")
    quality, A, B, k, reading, cards, witnesses = catalogue[idx]
    return {
        "A": A, "B": B, "k": k,
        "design_cards": cards,
        "logical_witness_upper_bound": reading,
        "logical_witnesses": witnesses,
        "complete_kernel_floor": 5,
        "selection_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without mutating the screened portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct relative-order cards provide distinct Dic7 codes: the measured sixty-slot parent collapsed to thirteen distinct codes, with multiple primary cards marked iso_dup.
# REFUTED: The normalized-direction catalogue supplies twenty evaluator-distinct Dic7 codes: its sixty measured slots retained fourteen classes, and all five finalists remained at k=252,d30s=3.
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
