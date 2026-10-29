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
CONCEPT = ("GL(2,3), order 48, shapes (1,2)x(3,4): a noncommuting quartet/pentad A component and independently optimized three-check-row B component should remove the parent's measured weight-twelve bottleneck and improve fitness in cell 0.100.")
GROUP = ("Keep the exact GL(2,F3) construction, encoding and order 48. Its (1,2)x(2,5) bed reached fitness 39.0, but multiple quartet-based programs retained five d30s=12 readings. The supplied BEDS list has not evaluated this group at (1,2)x(3,4). No distance transfer is assumed.")
SHAPES = ("Target cell 0.100 with (1,2)x(3,4): n=11*48=528, kb=48 and guaranteed rate 1/11. The shape allows q<=54 and retains q=48. Components have 96 variables/48 checks and 192 variables/144 checks. Current-parent minima were entirely sector one, spanning both A columns and multiple B columns, with no Arow/Brow minima. Exact lifted ranks require k>=48 and 8*k<n.")
FILLING = ("Twenty canonical weighted B architectures have column term totals three and row totals 3..7. A counts are (4,5), so HX=9+3=12 and HZ<=7+5=12. Independently optimize determinant-controlled central-pair-free A cells and order-six/eight B binomials using verified component syndromes. Retain short witnesses across independent rescreening, verify logical embeddings modulo stabilizers, and search stabilizer dressings. Exact logical-four exclusion and unequal uncoloured Tanner signatures gate emitted codes. Restricted exclusions and sampled witness readings do not certify distance.")
FAMILY_NAME = "r010_GL2F3_12x34_pentad_canonical_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # GL(2,3): (a,b,c,d) encodes [[a,b],[c,d]] over F3.
    if not valid_q(q):
        raise ValueError("GL(2,3) requires order 48")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = g
        t = (a*d-b*c) % 3
        # Both nonzero elements of F3 are their own reciprocals.
        return ((t*d) % 3, (-t*b) % 3,
                (-t*c) % 3, (t*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)

from functools import lru_cache

# Negative-determinant term quotas for the quartet and pentad.
_A_DET_CARDS = ((1,1), (1,2), (2,2), (2,3), (3,4))


@lru_cache(maxsize=1)
def _low_supports():
    """Twenty intrinsic architectures, canonical under B row/column swaps."""
    from itertools import product, permutations, combinations_with_replacement
    types = tuple(c for c in product(range(4), repeat=3) if sum(c) == 3)
    perms = tuple(permutations(range(3)))
    cards = set()
    for columns in combinations_with_replacement(types, 4):
        totals = tuple(sum(c[i] for c in columns) for i in range(3))
        if min(totals) < 3 or max(totals) > 7:
            continue
        canonical = min(
            tuple(sorted(tuple(c[i] for i in p) for c in columns))
            for p in perms)
        cards.add(canonical)

    def priority(columns):
        entries = tuple(x for c in columns for x in c)
        totals = tuple(sum(c[i] for c in columns) for i in range(3))
        return (entries.count(3), max(totals)-min(totals),
                entries.count(0), columns)

    ordered = sorted(cards, key=priority)
    if len(ordered) < 20:
        raise ValueError("fewer than twenty canonical low-rate B supports")
    result = []
    for columns in ordered[:20]:
        sb = tuple(tuple(c[i] for c in columns) for i in range(3))
        assert _grid_weights([[4, 5]], sb) == 12
        assert _no_dead_lines([[4, 5]], sb)
        result.append(sb)
    return tuple(result)


def _determinant(g):
    return (g[0] * g[3] - g[1] * g[2]) % 3


def _profile_cell(rng, pools, count, negative):
    """Disjoint determinant classes ensure distinct terms."""
    return (rng.sample(pools[1], count - negative)
            + rng.sample(pools[2], negative))


def _extract_components(hx, hz, q):
    """Extract both component maps from the scaffold's actual lifts."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(
                ((row >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca)))
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _short_kernel(cols, through_six=False):
    """Exact component screening through four, optionally through six.

    Sentinel 5/7 means no witness through four/six, not a distance
    estimate. After excluding weights <=4, triple collisions are
    disjoint and each weight-six word has ten triple partitions.
    """
    if 0 in cols:
        return (1, 1)
    if len(set(cols)) != len(cols):
        return (2, 1)
    low = {s: 1 for s in cols}
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            previous = low.get(s, 0)
            if previous:
                return (3 if previous == 1 else 4, 1)
            low[s] = 2
    if not through_six:
        return (5, 0)

    seen = {}
    hits = 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if not s:
                    return (3, 1)
                if s in low:
                    return (5, 1)
                previous = seen.get(s, 0)
                hits += previous
                seen[s] = previous + 1
    return (6, hits // 10) if hits else (7, 0)


def _kernel_probe(cols, rng, trials):
    """Randomized classical kernel witnesses: upper bounds only."""
    width = len(cols)
    best = width + 1
    for _ in range(trials):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        deps = []
        for c in order:
            s, v = cols[c], 1 << c
            while s:
                p = s.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s, v)
                    break
                s ^= old[0]
                v ^= old[1]
            if not s:
                deps.append(v)
                best = min(best, v.bit_count())

        for _ in range(2):
            changed = False
            for a in range(len(deps)):
                v = deps[a]
                weight = v.bit_count()
                for b in range(len(deps)):
                    if a == b:
                        continue
                    z = v ^ deps[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                deps[a] = v
                best = min(best, weight)
            if not changed:
                break
    return best


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_4(checks, stabilizers, width):
    """Exact quantum-logical existence test through weight four.

    Equal syndromes of supports of size <=2 cover every commuting
    support through four. One representative suffices: if a collision
    is a stabilizer, it preserves the quotient class of that syndrome.
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

    cols = _component_columns(checks, width)
    seen = {0: 0}
    bits = [1 << i for i in range(width)]
    for i, s in enumerate(cols):
        old = seen.get(s)
        if old is None:
            seen[s] = bits[i]
        elif outside(bits[i] ^ old):
            return True

    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            v = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = v
            elif outside(v ^ old):
                return True
    return False


def _low_words(cols, rng, trials, seeds=()):
    """Retain genuine kernel supports across independent information sets."""
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))
    for trial in range(trials):
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

        for sweep in range(2):
            reducers = sorted(set(dependencies+memory), key=int.bit_count)[:32]
            for i, word in enumerate(dependencies):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                dependencies[i] = word

        short = sorted(set(dependencies+memory), key=int.bit_count)[:40]
        discoveries = set(short+memory)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    discoveries.add(combined)
        if trials >= 12 and trial % 4 == 0:
            small = short[:8]
            for i, word in enumerate(small):
                for j in range(i):
                    for h in range(j):
                        combined = word ^ small[j] ^ small[h]
                        if combined:
                            discoveries.add(combined)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
    return memory


def _low_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _low_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _low_dressed(item, q):
    """Verify component embeddings and search single/pair stabilizer dressings."""
    mask = (1 << q)-1
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    best = item["n"]+1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], item["hx"], item["hz"]),
            ("B", item["bw"], item["hz"], item["hx"])):
        span = _low_basis(stabilizers)
        logicals = []
        for seed in seeds[:16]:
            for fixed in range(cb if side == "A" else ca):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((j*cb+fixed)*q) for j in range(ca))
                else:
                    word = seed << (fixed*cb*q)
                # Classify only actual embeddings of the verified maps.
                if any((word & row).bit_count() & 1 for row in checks):
                    raise ValueError("verified component embedding failed")
                if _low_outside(word, span):
                    best = min(best, word.bit_count())
                    logicals.append(word)

        for word in sorted(set(logicals), key=int.bit_count)[:8]:
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = []
                for row in stabilizers:
                    cw = (word ^ row).bit_count()
                    nearby.append((cw, row))
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                nearby.sort(key=lambda pair: pair[0])
                moves = [row for _, row in nearby[:12]]
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        combined = word ^ row ^ other
                        cw = combined.bit_count()
                        if cw < weight:
                            chosen, weight = combined, cw
                if chosen == word:
                    break
                word = chosen
                best = min(best, weight)
    return best


def _low_signature(hx, hz, n):
    """Uncoloured Tanner invariant: unequal signatures prove nonisomorphism.

    Equal signatures are conservatively rejected, not called isomorphic.
    Neither group labels nor the protograph ordering enters the signature.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0]*len(rows)
    local = [Counter() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                local[i][(weights[j], overlap)] += 1
                local[j][(weights[i], overlap)] += 1
    profiles = []
    for i, neighbors in enumerate(adjacency):
        triangles, remaining = 0, neighbors
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        profiles.append((weights[i], tuple(sorted(local[i].items())),
                         triangles//2))
    degrees = _component_columns(rows, n)
    return (tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(c.bit_count() for c in degrees).items())))


@lru_cache(maxsize=1)
def _low_bank(G):
    """Pure deterministic bank of independently optimized, invariant-distinct cards."""
    q = G.q
    assert q == 48
    pools = {d: [g for g in G.elements if _determinant(g) == d]
             for d in (1, 2)}
    orders = {}
    for g in G.elements:
        value = G.e
        for t in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = t
                break
    palettes = {t: [g for g in G.elements if orders[g] == t]
                for t in (6, 8)}
    central = (2, 0, 0, 2)
    stub_a = [[[G.e] for _ in range(A_SHAPE[1])]]
    stub_b = [[[G.e] for _ in range(B_SHAPE[1])]
              for _ in range(B_SHAPE[0])]
    bank, emitted = [], set()

    def reading(words, width):
        return min((w.bit_count() for w in words), default=width+1)

    for slot in range(N_SLOTS):
        card = slot % 20
        sb = [list(row) for row in _low_supports()[card]]
        sa = [[4, 5]]
        negative = _A_DET_CARDS[card % len(_A_DET_CARDS)]
        pair_order = (6, 8)[(card//5 + slot//20) % 2]
        rng = random.Random(BASE_SEED*10007 + 480103401 + 100003*slot)
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)

        def multiplet(count, nf):
            for retry in range(192):
                cell = _profile_cell(rng, pools, count, nf)
                if any(G.mul(G.inv(g), h) == central
                       for i, g in enumerate(cell) for h in cell[:i]):
                    continue
                relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                if not any(G.mul(g, h) != G.mul(h, g)
                           for i, g in enumerate(relatives)
                           for h in relatives[:i]):
                    continue
                return cell
            raise ValueError("noncommuting determinant multiplet exhausted")

        def proposal(side, attempt):
            if side == "A":
                return [[multiplet(sa[0][j], negative[j])
                         for j in range(A_SHAPE[1])]]
            grid = []
            for i, row in enumerate(sb):
                out = []
                for j, count in enumerate(row):
                    if count == 0:
                        cell = []
                    elif count == 1:
                        cell = [G.pick(rng)]
                    elif count == 2:
                        base = G.pick(rng)
                        cell = [base, G.mul(base, rng.choice(palettes[pair_order]))]
                    else:
                        nf = 1 + ((card+i+j+attempt//12) % 2)
                        cell = multiplet(count, nf)
                    out.append(cell)
                grid.append(out)
            return grid

        def component_pool(side, quota):
            candidates, seen = [], set()
            width = (A_SHAPE[1] if side == "A" else B_SHAPE[1])*q
            for attempt in range(quota):
                grid = proposal(side, attempt)
                labels = tuple(tuple(sorted(c)) for row in grid for c in row)
                if labels in seen:
                    continue
                seen.add(labels)
                A, B = (grid, stub_b) if side == "A" else (stub_a, grid)
                hx, hz, _ = lifted_matrices(G, A, B)
                ar, br = _extract_components(hx, hz, q)
                rows = ar if side == "A" else br
                cols = _component_columns(rows, width)
                flag, _ = _short_kernel(cols)
                if flag < 5:
                    continue
                words = _low_words(cols, rng, 2)
                item = dict(grid=grid, cols=cols, words=words,
                            d=reading(words, width),
                            cycles=_component_four_cycles(cols))
                candidates.append(item)
                candidates.sort(
                    key=lambda x: (x["d"], -x["cycles"]), reverse=True)
                del candidates[6:]

            for item in candidates:
                item["words"] = _low_words(
                    item["cols"], rng, 24 if side == "A" else 16,
                    item["words"])
                item["d"] = reading(item["words"], width)
            candidates.sort(
                key=lambda x: (x["d"], -x["cycles"]), reverse=True)
            return candidates[:4]

        chosen = None
        for batch in range(2):
            apool = component_pool("A", 48+24*batch)
            bpool = component_pool("B", 48+24*batch)
            finalists = []
            for ai in apool:
                for bi in bpool:
                    A, B = ai["grid"], bi["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (n == 528 and k >= q and 8*k < n):
                        continue
                    d = min(ai["d"], bi["d"])
                    item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                                ac=ai["cols"], bc=bi["cols"],
                                aw=list(ai["words"]), bw=list(bi["words"]),
                                da=ai["d"], db=bi["d"],
                                cycles=ai["cycles"]+bi["cycles"])
                    finalists.append((k*d*d, d, -item["cycles"], item))
            finalists.sort(key=lambda x: x[:3], reverse=True)

            deep = []
            for _, _, _, item in finalists[:4]:
                # Independent searches retain, rather than erase, old witnesses.
                item["aw"] = _low_words(item["ac"], rng, 40, item["aw"])
                item["bw"] = _low_words(item["bc"], rng, 32, item["bw"])
                item["da"] = min(item["da"], reading(item["aw"], 2*q))
                item["db"] = min(item["db"], reading(item["bw"], 4*q))
                logical = _low_dressed(item, q)
                if logical > item["n"]:
                    continue
                d = min(item["da"], item["db"], logical)
                key = (item["k"]*d*d, d, logical, -item["cycles"])
                deep.append((key, item))
            deep.sort(key=lambda x: x[0], reverse=True)

            for key, item in deep:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _small_logical_4(hx, hz, n):
                    continue
                if _small_logical_4(hz, hx, n):
                    continue
                signature = _low_signature(hx, hz, n)
                if signature in emitted:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert item["k"] >= q and 8*item["k"] < n
                assert all(len(c) == len(set(c)) <= CELL_MAX
                           for grid in (item["A"], item["B"])
                           for row in grid for c in row)
                emitted.add(signature)
                chosen = dict(cells={"A": item["A"], "B": item["B"]},
                              k=item["k"], selection=key)
                break
            if chosen is not None:
                break
        # Failed cards never substitute an emitted orbit or archived filling.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the slot's canonical architecture and fresh screened labels."""
    selected = _low_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh low-rate pentad filling passed screening")
    return {"idx": idx, "cells": selected["cells"],
            "k": selected["k"], "selection": selected["selection"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: C9 monomial voltage rows with isolated odd trinomial upgrades and exact weight<=3 logical rejection did not exceed distance four: all five measured parent codes had k=283 and d30s=4, with minima entirely in sector 1.
# REFUTED: GL(2,3) quartet improvements using noncommuting relative labels, central-pair screening and rank-bonus selection remove the weight-twelve bottleneck; the shown reflection-budget program still measured five d30s=12 codes.
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
