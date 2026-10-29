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
CONCEPT = ("Dic9, order 36, A=(2,3), B=(2,4): independently optimized two-check components, cyclic-subgroup witness screening and stabilizer-aware joint selection should exceed the 0.150 frontier fitness 37.4; measured evaluation can refute this.")
GROUP = ("Jump from GL(2,3) to Dic9, x^18=1, y^2=x^9, y*x*y^-1=x^-1. Its unique involution makes it genuinely different from GL(2,3). RUN STATE's order-36 (2,3)x(2,4) bed reached 37.4, and the Dic12 parent-shape bed reached 43.5. These motivate this order and family without identifying the hashed order-36 group or assuming distance transfer.")
SHAPES = ("Target cell 0.150 rather than the parent's 0.250: n=16*36=576, kb=72, guaranteed rate 1/8, maximum lift order 37. Component variables/checks are 108/72 and 144/72. Parent minima broadly occupied sector-one Acol/Bcol blocks, with one mixed-sector weight-eleven X witness. Give both components two check rows and enforce assembled 0.125<=k/n<0.175.")
FILLING = ("Twenty deliberate tasks cross five support architectures with four coset-occupancy rules. Every row and column is occupied; both CSS row maxima are <=12. Exclude central-involution-separated terms, vary long rotation binomials versus outer-coset binomials, and independently optimize verified component maps across rank strata. Rescreen cyclic-subgroup norm witnesses and randomized kernels, then test full logicals modulo stabilizers and exclude logical weights <=4 exactly. A combined uncoloured incidence invariant rejects emitted orbit copies. Restricted searches and randomized readings do not certify distance.")
FAMILY_NAME = "r015_Dic9_23x24_two_check_norm_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # Dic9: (a,b)=x^a y^b, x^18=1, y^2=x^9.
    if not valid_q(q):
        raise ValueError("Dic9 has order 36")
    els = [(a,b) for b in range(2) for a in range(18)]

    def red(g):
        a,b = g
        return ((a + 9*(b//2)) % 18, b % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (c if b == 0 else -c) + 9*b*d) % 18,
                (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-a) % 18, 0) if b == 0 else ((a+9) % 18, 1)

    def idx(g):
        a,b = red(g)
        return 18*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


# Entries describe B columns as (top count, bottom count).
# A is always (4,4), hence row sum 8 and column sums 4.
# B rows are <=8 and columns <=4: wx<=8+4, wz<=8+4.
_SUPPORT_PLANS = (
    ((2,1), (2,1), (1,2), (1,2), (2,2)),
    ((3,0), (2,1), (1,2), (1,2), (1,3)),
    ((3,0), (0,3), (2,1), (1,2), (2,2)),
    ((2,1), (2,1), (2,1), (1,2), (1,3)),
    ((2,1), (2,1), (1,2), (1,2), (1,2)),
)
_ORDER_CARDS = ((3,4), (4,8), (6,8), (3,8))


def _element_orders(G):
    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("element order did not divide group order")
    return orders


def _designed_grid(rng, G, counts, mode, side, orders):
    """Independent fillings with prescribed relative-order structure."""
    palettes = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (3,4,6,8)
    }
    result = []
    for i, sizes in enumerate(counts):
        row = []
        for j, weight in enumerate(sizes):
            if weight <= 1:
                row.append([] if weight == 0 else [G.pick(rng)])
                continue
            wanted = (_ORDER_CARDS[mode][j % 2] if side == "A"
                      else (3,4,6,8)[(mode + i + j) % 4])
            selected = None
            for attempt in range(80):
                anchor = G.pick(rng)
                cell = [anchor, G.mul(anchor, rng.choice(palettes[wanted]))]
                while len(cell) < weight:
                    legal = [
                        g for g in G.elements
                        if g not in cell and all(
                            orders[G.mul(G.inv(h), g)] > 2 for h in cell)
                    ]
                    if not legal:
                        break
                    cell.append(rng.choice(legal))
                if len(cell) != weight:
                    continue
                relatives = [G.mul(G.inv(anchor), g) for g in cell[1:]]
                if weight >= 3 and not any(
                    G.mul(g,h) != G.mul(h,g)
                    for t, g in enumerate(relatives)
                    for h in relatives[:t]
                ):
                    continue
                selected = cell
                break
            if selected is None:
                raise ValueError("relative-order cell recipe failed")
            row.append(selected)
        result.append(row)
    return result


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
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
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _no_kernel_le6(columns):
    """Exact component test, not a quantum-distance certificate.

    Every support of weight <=6 partitions into two different subsets
    of size <=3. A syndrome collision therefore detects such a support.
    Only existence is tested; no witness census is claimed.
    """
    seen = {0}
    for syndrome in columns:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _component_data(hx, hz, q):
    """Extract arbitrary-shaped components from the VERIFIED full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i*cb)*q+t]
            ar.append(sum(
                ((row >> (j*cb*q)) & mask) << (j*q)
                for j in range(ca)))
    br = [row & ((1 << (cb*q))-1) for row in hz[:rb*q]]
    return ar, br, _columns(ar, ca*q), _columns(br, cb*q)


def _unused_gl2_parent_policy(rng, G, idx):
    """Retained parent policy; the Dic9 NEW_CTX below never calls it."""
    from itertools import combinations

    q = G.q
    recipe, mode = idx % 20, idx % 4
    incidence = _SUPPORT_PLANS[recipe // 4]
    sa = [[4, 4]]
    sb = [[column[i] for column in incidence] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    els = G.elements
    identity = G.idx(G.e)
    multiplication = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    inverses = [G.idx(G.inv(g)) for g in els]
    orders = _element_orders(G)
    pools = {
        d: [g for g in els if (g[0]*g[3] - g[1]*g[2]) % 3 == d]
        for d in (1, 2)
    }

    def closure(generators, cap):
        """Group operations construct label constraints, never another lift."""
        members = [identity]
        seen = {identity}
        for x in members:
            for g in generators:
                y = multiplication[x][g]
                if y not in seen:
                    seen.add(y)
                    members.append(y)
                    if len(members) > cap:
                        return None
        return tuple(sorted(members))

    def draw(size, negative, full_group=False):
        if not size:
            return []
        if size == 1:
            return [rng.choice(pools[1 + negative])]
        for attempt in range(192):
            cell = (rng.sample(pools[1], size-negative)
                    + rng.sample(pools[2], negative))
            positions = [G.idx(g) for g in cell]
            bad = False
            for a, g in enumerate(positions):
                for h in positions[:a]:
                    relative = multiplication[inverses[h]][g]
                    if orders[els[relative]] <= 2:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            anchor = positions[0]
            relatives = [
                multiplication[inverses[anchor]][g]
                for g in positions[1:]
            ]
            if size == 2:
                if orders[els[relatives[0]]] not in (6, 8):
                    continue
            elif full_group:
                generated = closure(relatives, q)
                if generated is None or len(generated) != q:
                    continue
            elif not any(
                multiplication[g][h] != multiplication[h][g]
                for a, g in enumerate(relatives) for h in relatives[:a]
            ):
                continue
            return cell
        raise ValueError("full-generation occupancy recipe exhausted")

    def basis(rows):
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

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    mask = (1 << q) - 1

    def embed_a(word):
        return tuple(
            ((word & mask) << (beta*q))
            | ((word >> q) << ((5+beta)*q))
            for beta in range(5)
        )

    def embed_b(word):
        return (word, word << (5*q))

    def probe(cols, embed, stabilizers, trials, initial=None,
              generators=()):
        """Only actual non-stabilizer witnesses lower the returned reading."""
        best = len(cols)+1 if initial is None else initial
        order = list(range(len(cols)))
        memory = []

        def observe(word):
            nonlocal best
            weight = word.bit_count()
            if word and weight < best and any(
                outside(full, stabilizers) for full in embed(word)
            ):
                best = weight

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
                    observe(word)

            words = sorted(set(dependencies + memory),
                           key=int.bit_count)[:48]
            for sweep in range(2):
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for other in words:
                        trial_word = word ^ other
                        tw = trial_word.bit_count()
                        if 0 < tw < weight:
                            word, weight = trial_word, tw
                    reduced.append(word)
                    observe(word)
                words = sorted(set(reduced), key=int.bit_count)

            discoveries = set(words)
            for a, word in enumerate(words):
                for other in words[:a]:
                    combined = word ^ other
                    if combined:
                        observe(combined)
                        discoveries.add(combined)

            # Full-code searches also descend actual logical seeds using
            # stabilizer generators, preserving their nonzero classes.
            if generators:
                logical_seeds = [
                    word for word in dependencies
                    if outside(word, stabilizers)
                ]
                logical_seeds.sort(key=int.bit_count)
                for word in logical_seeds[:8]:
                    for sweep in range(2):
                        changed = False
                        for row in generators:
                            combined = word ^ row
                            if combined.bit_count() < word.bit_count():
                                word = combined
                                changed = True
                        if not changed:
                            break
                    observe(word)
                    discoveries.add(word)
            memory = sorted(discoveries, key=int.bit_count)[:48]
            if best <= 4:
                break
        return best

    # Construct small subgroup partitions once for this slot. Both coset
    # orientations are included; all their syndromes come from verified
    # component columns, not from a separately implemented matrix action.
    subgroups = set()
    for a in range(q):
        subgroup = closure((a,), 12)
        if subgroup is not None and len(subgroup) in (2,3,4,6,8,12):
            subgroups.add(subgroup)
        for b in range(a):
            subgroup = closure((a, b), 12)
            if subgroup is not None and len(subgroup) in (2,3,4,6,8,12):
                subgroups.add(subgroup)

    partitions = set()
    for subgroup in subgroups:
        for orientation in range(2):
            unseen = set(range(q))
            fibers = []
            while unseen:
                g = min(unseen)
                coset = {
                    multiplication[g][h] if orientation == 0
                    else multiplication[h][g]
                    for h in subgroup
                }
                unseen.difference_update(coset)
                fibers.append(sum(1 << h for h in coset))
            partitions.add((len(subgroup), tuple(sorted(fibers))))
    partitions = sorted(partitions, reverse=True)

    def structured(cols, embed, stabilizers, cutoff, only_two=False):
        """Exhaustively test bounded unions within each listed partition.

        This is a restricted logical search, not an exhaustive quantum
        distance test. One syndrome representative suffices: a collision
        that is a stabilizer preserves the representative's quotient class.
        """
        best = len(cols)+1
        found = set()
        for size, fibers in partitions:
            if (size == 2) != only_two:
                continue
            limit = cutoff // size
            if not limit:
                continue
            words = [
                fiber << (block*q)
                for block in range(len(cols)//q) for fiber in fibers
            ]
            syndromes = []
            for word in words:
                syndrome = 0
                while word:
                    bit = word & -word
                    syndrome ^= cols[bit.bit_length()-1]
                    word ^= bit
                syndromes.append(syndrome)

            seen = {0: 0}
            radius = (limit+1)//2
            for weight in range(1, radius+1):
                store = not (limit % 2 and weight == radius)
                for indices in combinations(range(len(words)), weight):
                    syndrome, word = 0, 0
                    for c in indices:
                        syndrome ^= syndromes[c]
                        word ^= words[c]
                    old = seen.get(syndrome)
                    if old is not None:
                        witness = word ^ old
                        ww = witness.bit_count()
                        if witness and ww <= cutoff and any(
                            outside(full, stabilizers)
                            for full in embed(witness)
                        ):
                            best = min(best, ww)
                            if ww == 12:
                                found.add(witness)
                    elif store:
                        seen[syndrome] = word
            if best <= 8:
                break
        return best, len(found)

    # New cards prescribe determinant occupancies, not translations of
    # the parent's relative-order labeling. Each A cell generates GL(2,3)
    # through its anchor-relative elements.
    a_negative = ((1,1), (1,2), (2,2), (2,3))[mode]
    strata = {}
    for attempt in range(72 + 8*(idx//20)):
        A = [[draw(4, a_negative[j], True) for j in range(2)]]
        B = []
        for i in range(2):
            row = []
            for j in range(5):
                count = sb[i][j]
                if count <= 1:
                    negative = (mode+i+j+attempt) % 2 if count else 0
                elif count == 2:
                    negative = (mode+i+j) % 3
                else:
                    negative = 1 + ((mode+i+j+attempt//12) % (count-1))
                row.append(draw(count, negative))
            B.append(row)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (225*n <= 1000*k < 275*n):
            continue
        ar, br, ac, bc = _component_data(hx, hz, q)
        floor = min(_short_floor(ac), _short_floor(bc))
        cycles = _component_cycles(ar) + _component_cycles(br)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ac, bc=bc, floor=floor, cycles=cycles)
        bucket = strata.setdefault(k, [])
        bucket.append(item)
        bucket.sort(key=lambda c: (c["floor"], -c["cycles"]),
                    reverse=True)
        del bucket[4:]

    raw = [item for bucket in strata.values() for item in bucket]
    if not raw:
        raise ValueError("no full-generation filling in cell 0.250")
    raw.sort(key=lambda c: (c["floor"], c["k"], -c["cycles"]),
             reverse=True)

    def merit(item):
        d = min(item["da"], item["db"], item.get("full", item["n"]+1))
        return (item["floor"], item["k"]*d*d/item["n"], d,
                -item["hits"], item["da"]+item["db"], -item["cycles"])

    pool = []
    for item in raw[:12]:
        item["xs"], item["zs"] = basis(item["hx"]), basis(item["hz"])
        da = probe(item["ac"], embed_a, item["zs"], 4)
        db = probe(item["bc"], embed_b, item["xs"], 6)
        if da > len(item["ac"]) or db > len(item["bc"]):
            # Never promote a no-witness sentinel as a large distance.
            continue
        wa, ha = structured(item["ac"], embed_a, item["zs"], 12)
        wb, hb = structured(item["bc"], embed_b, item["xs"], 12)
        item.update(da=min(da, wa), db=min(db, wb), hits=ha+hb)
        pool.append(item)

    if not pool:
        # A legal fallback is preferable to voiding the structural card.
        winner = raw[0]
    else:
        finalists = sorted(pool, key=merit, reverse=True)[:4]
        for item in finalists:
            item["da"] = probe(
                item["ac"], embed_a, item["zs"], 24, item["da"])
            item["db"] = probe(
                item["bc"], embed_b, item["xs"], 32, item["db"])
            wa, ha = structured(
                item["ac"], embed_a, item["zs"], 12, only_two=True)
            wb, _ = structured(
                item["bc"], embed_b, item["xs"], 8, only_two=True)
            item["da"], item["db"] = min(item["da"], wa), min(item["db"], wb)
            item["hits"] += ha

            # These verified full-matrix probes include sector 2 and
            # mixed-sector supports, with exact stabilizer membership.
            dx = probe(
                _columns(item["hz"], item["n"]), lambda w: (w,),
                item["xs"], 8, item["db"], item["hx"])
            dz = probe(
                _columns(item["hx"], item["n"]), lambda w: (w,),
                item["zs"], 8, item["da"], item["hz"])
            item["full"] = min(dx, dz)
        winner = max(finalists, key=merit)

    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 225*winner["n"] <= 1000*winner["k"] < 275*winner["n"]
    assert all(len(cell) == len(set(cell))
               for grid in (winner["A"], winner["B"])
               for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": winner["A"], "B": winner["B"]}}


# These support cards supersede the unused GL-only support policies.
# Listed maxima (HX,HZ): (10,12), (10,12), (10,12), (11,12), (10,12).
# Each entry contains the complete A and B term-count grids.
_D9_SUPPORTS = (
    (((0,3,3), (4,1,1)), ((1,2,2,3), (3,2,2,1))),
    (((2,2,2), (2,2,2)), ((3,3,1,1), (1,1,3,3))),
    (((1,2,3), (3,2,1)), ((0,2,3,3), (4,2,1,1))),
    (((3,2,2), (1,2,2)), ((3,2,2,1), (1,2,2,3))),
    (((2,2,2), (1,1,1)), ((3,2,2,2), (1,2,2,2))),
)


def _d9_draw(rng, G, counts, mode, side, phase):
    """Prescribed intrinsic occupancies, with independent internal labels."""
    from math import gcd
    pools = [[g for g in G.elements if g[1] == b] for b in (0,1)]
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, count in enumerate(row):
            if count == 0:
                out.append([])
                continue
            if count == 1:
                out.append([rng.choice(pools[(phase+i+j) % 2])])
                continue
            if count == 2:
                if mode == 0:
                    outer = (i+j+int(side == "B")) % 3
                elif mode == 1:
                    outer = 2*((i+j) % 2)
                elif mode == 2:
                    outer = 1
                else:
                    outer = 2*((i+2*j+phase) % 2)
            elif mode == 0:
                outer = count//2 + (count % 2)*((i+j) % 2)
            elif mode == 1:
                outer = 1
            elif mode == 2:
                outer = count-1
            else:
                outer = 1 + ((i+j+int(side == "B")) % (count-1))

            quota = [0]*(count-outer) + [1]*outer
            selected = None
            for restart in range(128):
                rng.shuffle(quota)
                cell = []
                for b in quota:
                    legal = []
                    for g in pools[b]:
                        if g in cell:
                            continue
                        if any(G.mul(G.inv(h), g) == (9,0)
                               for h in cell):
                            continue
                        if count == 2 and cell and b == cell[0][1]:
                            order = 18 // gcd(
                                (g[0]-cell[0][0]) % 18, 18)
                            wanted = 9 if mode == 3 else 18
                            if order != wanted:
                                continue
                        legal.append(g)
                    if not legal:
                        break
                    cell.append(rng.choice(legal))
                if len(cell) == count:
                    selected = cell
                    break
            if selected is None:
                raise ValueError("Dic9 occupancy recipe exhausted")
            out.append(selected)
        grid.append(out)
    return grid


def _d9_basis(rows):
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


def _d9_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d9_low4(checks, stabilizers, n):
    """Exact existence test for logical weights 1..4, modulo stabilizers."""
    span = _d9_basis(stabilizers)
    cols = _columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _d9_outside(word ^ old, span):
            return True
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _d9_outside(word ^ old, span):
                return True
    return False


def _d9_norm_words(cols, fibers, q, deep=False):
    """Actual kernel witnesses in restricted cyclic-subgroup subspaces.

    The disjoint fiber indicators are constructed with group operations.
    Their syndromes come only from verified component columns. This is
    neither an unrestricted distance test nor an exhaustive witness census.
    """
    from itertools import combinations
    found = set()
    for h in ((2,3,6,9,18) if deep else (3,6,9,18)):
        words = [fiber << (block*q)
                 for block in range(len(cols)//q)
                 for fiber in fibers[h]]
        syndromes = []
        for original in words:
            word, syndrome = original, 0
            while word:
                bit = word & -word
                syndrome ^= cols[bit.bit_length()-1]
                word ^= bit
            syndromes.append(syndrome)
        seen = {0: 0}
        radius = 3 if h == 2 and deep else 2
        for size in range(1, radius+1):
            for indices in combinations(range(len(words)), size):
                syndrome, coefficient = 0, 0
                for c in indices:
                    syndrome ^= syndromes[c]
                    coefficient |= 1 << c
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = coefficient
                    continue
                dependency = coefficient ^ old
                if not dependency or h*dependency.bit_count() > 24:
                    continue
                expanded = 0
                while dependency:
                    bit = dependency & -dependency
                    expanded ^= words[bit.bit_length()-1]
                    dependency ^= bit
                if expanded:
                    found.add(expanded)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:24])


def _d9_logical_probe(checks, stabilizers, n, rng, trials, seeds=()):
    """Find genuine logical witnesses, with sparse stabilizer dressings."""
    span = _d9_basis(stabilizers)
    cols = _columns(checks, n)
    best, retained = n+1, set()

    def observe(word):
        nonlocal best, retained
        if not word or not _d9_outside(word, span):
            return
        weight = word.bit_count()
        if weight < best:
            best, retained = weight, {word}
        elif weight == best:
            retained.add(word)

    for word in seeds:
        if not any((word & row).bit_count() & 1 for row in checks):
            observe(word)

    order, memory = list(range(n)), []
    for trial in range(trials):
        rng.shuffle(order)
        # Alternate information sets that expose either LP sector first.
        if trial % 3 != 2:
            split = A_SHAPE[1]*B_SHAPE[1]*36
            order.sort(key=lambda c:
                       (c < split) if trial % 3 == 0 else (c >= split))
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
                observe(word)

        words = sorted(set(dependencies+memory), key=int.bit_count)[:48]
        raw = list(words)
        for sweep in range(2):
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in words[:32]:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                observe(word)
                reduced.append(word)
            words = sorted(set(reduced), key=int.bit_count)
        discoveries = set(words)
        for a, word in enumerate(words[:32]):
            for other in words[:a]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        logical = [w for w in raw if _d9_outside(w, span)]
        logical += list(retained)
        for word in sorted(set(logical), key=int.bit_count)[:6]:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda item: item[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:12]]
                for a, row in enumerate(moves):
                    for other in moves[:a]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
                observe(word)
            discoveries.add(word)
        memory = sorted(discoveries, key=int.bit_count)[:48]
        if best <= 4:
            break
    return best, tuple(sorted(retained)[:12])


def _d9_incidence_signature(hx, hz, n):
    """Different signatures certify different uncoloured Tanner graphs.

    Equality is conservatively rejected; it does not prove isomorphism.
    X/Z check names and group labels are deliberately absent.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    profiles = [Counter() for _ in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] += 1
            if overlap:
                profiles[i][(weights[j], overlap)] += 1
                profiles[j][(weights[i], overlap)] += 1
    local = Counter(
        (weights[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles))
    degrees = Counter(col.bit_count() for col in _columns(rows, n))
    return (n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(degrees.items())),
            tuple(sorted(overlaps.items())), tuple(sorted(local.items())))


@__import__("functools").lru_cache(maxsize=1)
def _d9_design_bank(G):
    """Pure deterministic bank of independently optimized structural tasks."""
    q = G.q
    assert q == 36
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    zero_a = [[[] for _ in range(ca)] for _ in range(ra)]
    zero_b = [[[] for _ in range(cb)] for _ in range(rb)]

    fibers = {}
    for h in (2,3,6,9,18):
        unseen, partition = set(range(q)), []
        while unseen:
            g = G.elements[min(unseen)]
            orbit = {G.idx(G.mul(g, (t*(18//h), 0)))
                     for t in range(h)}
            assert len(orbit) == h
            unseen.difference_update(orbit)
            partition.append(sum(1 << t for t in orbit))
        fibers[h] = tuple(sorted(partition))

    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        sa, sb = _D9_SUPPORTS[geometry]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED*10007 + 36015073 + 100003*slot)

        def component_pool(side, quota):
            counts = sa if side == "A" else sb
            strata = {}
            seen = set()
            for attempt in range(quota):
                grid = _d9_draw(rng, G, counts, mode, side, slot+attempt)
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen:
                    continue
                seen.add(signature)
                A, B = ((grid, zero_b) if side == "A"
                        else (zero_a, grid))
                hx, hz, _ = lifted_matrices(G, A, B)
                ar, br, ac, bc = _component_data(hx, hz, q)
                rows, cols = (ar, ac) if side == "A" else (br, bc)
                rank = gf2_rank(rows)
                floor = _short_floor(cols)
                structured = _d9_norm_words(cols, fibers, q)
                d = _kernel_reading(cols, rng, 4)
                if structured:
                    d = min(d, structured[0].bit_count())
                if floor < 5:
                    d = min(d, floor)
                item = dict(grid=grid, cols=cols, rank=rank, floor=floor,
                            d=d, words=structured,
                            cycles=_component_cycles(rows))
                bucket = strata.setdefault(rank, [])
                bucket.append(item)
                bucket.sort(key=lambda x:
                            (x["floor"], min(x["d"], 24), -x["cycles"]),
                            reverse=True)
                del bucket[2:]
            pool = [x for bucket in strata.values() for x in bucket]
            pool.sort(key=lambda x:
                      (x["floor"], min(x["d"], 24), -x["cycles"], x["rank"]),
                      reverse=True)
            pool = pool[:8]
            for item in pool:
                item["words"] = _d9_norm_words(
                    item["cols"], fibers, q, deep=True)
                item["d"] = min(item["d"],
                                _kernel_reading(item["cols"], rng, 28))
                if item["words"]:
                    item["d"] = min(item["d"],
                                    item["words"][0].bit_count())
            pool.sort(key=lambda x:
                      (x["floor"], min(x["d"], 24), -x["cycles"], x["rank"]),
                      reverse=True)
            return pool[:4]

        winner = None
        for retry in range(2):
            ap = component_pool("A", 64+32*retry+4*(slot//20))
            bp = component_pool("B", 64+32*retry+4*(slot//20))
            proposals = []
            for ai in ap:
                for bi in bp:
                    A, B = ai["grid"], bi["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (125*n <= 1000*k < 175*n):
                        continue
                    d = min(ai["d"], bi["d"], 24)
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        ai=ai, bi=bi,
                        cycles=ai["cycles"]+bi["cycles"],
                        key=(min(ai["floor"], bi["floor"]), k*d*d,
                             -ai["cycles"]-bi["cycles"])))
            proposals.sort(key=lambda x: x["key"], reverse=True)
            finalists = []
            for item in proposals[:8]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _d9_low4(hx, hz, n) or _d9_low4(hz, hx, n):
                    continue
                fingerprint = _d9_incidence_signature(hx, hz, n)
                if fingerprint in fingerprints:
                    continue

                a_seeds = []
                for word in item["ai"]["words"]:
                    for beta in range(cb):
                        a_seeds.append(sum(
                            ((word >> (j*q)) & mask)
                            << ((j*cb+beta)*q) for j in range(ca)))
                b_seeds = [
                    word << (j*cb*q)
                    for word in item["bi"]["words"] for j in range(ca)]
                dx, wx = _d9_logical_probe(hz, hx, n, rng, 8, b_seeds)
                dz, wz = _d9_logical_probe(hx, hz, n, rng, 8, a_seeds)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx, dz=dz, wx=wx, wz=wz,
                            fingerprint=fingerprint)
                finalists.append(item)
                if len(finalists) >= 4:
                    break

            def merit(item):
                # Component readings are conservative selection heuristics:
                # their words can become stabilizers after embedding.
                d = min(item["dx"], item["dz"],
                        item["ai"]["d"], item["bi"]["d"], 24)
                return (item["k"]*d*d/item["n"], d,
                        min(item["dx"], item["dz"]), -item["cycles"])

            finalists.sort(key=merit, reverse=True)
            for item in finalists[:2]:
                dx, wx = _d9_logical_probe(
                    item["hz"], item["hx"], item["n"], rng, 20, item["wx"])
                dz, wz = _d9_logical_probe(
                    item["hx"], item["hz"], item["n"], rng, 20, item["wz"])
                item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
            if finalists:
                winner = max(finalists, key=merit)
                break

        if winner is None:
            bank.append(None)
            continue
        hx, hz = winner["hx"], winner["hz"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= W_MAX
        assert winner["k"] > 0
        assert all(len(cell) <= 5 and len(cell) == len(set(cell))
                   for grid in (winner["A"], winner["B"])
                   for row in grid for cell in row)
        fingerprints.add(winner["fingerprint"])
        bank.append(dict(
            cells={"A": winner["A"], "B": winner["B"]},
            k=winner["k"], card=(geometry, mode),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh, incidence-distinct Dic9 two-check design."""
    item = _d9_design_bank(G)[idx]
    if item is None:
        raise ValueError("Dic9 card found no fresh logical-screened 0.150 filling")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly selected cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting cell relatives and component dependency screens through six eliminate sector-1 weight-twelve logicals on GL(2,3), (1,2)x(2,5); all five measured parent codes retained both X and Z weight-twelve sector-1 witnesses.
# REFUTED: Full-group-generating A relatives and subgroup screening raise GL(2,3) five-code fitness above 38.25; the current measurements instead scored 37.7542, with four d30s=12 readings, one mixed-sector d30s=11 reading, and four occupancy-recipe failures.
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
