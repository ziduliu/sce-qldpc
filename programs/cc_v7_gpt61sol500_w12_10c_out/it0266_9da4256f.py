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
CONCEPT = ("C36, order 36, A=(2,3), B=(2,4): degree-four A columns and independently optimized commuting-cofactor B components should produce five measured codes whose mean score exceeds the 0.150 frontier fitness 41.1; measured evaluation can refute this.")
GROUP = ("Jump from nonabelian Dic9 to cyclic C36, a genuinely different group. C37 reached fitness 40.0 on these shapes and C46 reached 60.3 on equal two-check shapes, motivating an unlisted cyclic-order-36 bed without promising distance transfer. Cyclic subgroups and commuting minors supply genuine kernel supports; their syndromes and every rank are checked against scaffold-generated actions.")
SHAPES = ("Keep (2,3)x(2,4) for cell 0.150: n=576, kb=72, guaranteed rate 1/8 and maximum admissible lift order 37. Components have 108/72 and 144/72 variables/checks. Parent minima occupied sector-one Acol/Bcol blocks, broadly across B columns and often emphasizing Acol1. Exact full ranks retain only 0.125<=k/n<0.175; no component-rank formula substitutes for this check.")
FILLING = ("Twenty constructive tasks cross five B count profiles with four A gap profiles. A has six binomials, row sums six and column sums four. B rows sum to eight and columns have at most five terms, so HX<=11 and HZ<=12. Compile component actions only from verified unit lifts; select fresh voltages using syndrome-verified commuting minors, subgroup-indicator dependencies and retained kernel witnesses. Independently deepen every joint finalist, check actual logical membership and weights through four, and reject equal strengthened uncoloured Tanner invariants. Restricted and randomized readings remain upper bounds, never distance certificates.")
FAMILY_NAME = "r015_C36_23x24_gap_cofactor_subgroup_equaldeep_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # C36, encoded as powers of one generator.
    if not valid_q(q):
        raise ValueError("this cyclic bed requires order 36")
    els = [(a,) for a in range(36)]

    def red(g):
        return (g[0] % 36,)

    def mul(g, h):
        return ((g[0] + h[0]) % 36,)

    def inv(g):
        return ((-g[0]) % 36,)

    def idx(g):
        return g[0] % 36

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
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


# Five architectures crossed with four intrinsic C4 label policies.
# Maxima (HX,HZ): (10,12), (11,12), (10,12), (10,12), (10,12).
# Cards 2/3 spend the A-column budget reduction on denser B columns.
# Cards 0/1/4 retain promising degree-four geometries, but use fresh labels.
_D9_SUPPORTS = (
    (((2,2,2), (2,2,2)), ((3,3,1,1), (1,1,3,3))),
    (((3,2,2), (1,2,2)), ((3,2,2,1), (1,2,2,3))),
    (((1,2,1), (2,1,2)), ((2,2,2,3), (2,2,3,2))),
    (((1,2,1), (2,1,2)), ((1,3,2,3), (3,1,3,2))),
    (((0,3,3), (4,1,1)), ((1,2,2,3), (3,2,2,1))),
)


def _d9_draw(rng, G, counts, mode, side, phase):
    """C4 occupancy policies with long same-coset internal differences.

    chi(x^a y^b)=2*a+b mod 4 is used only to choose labels.
    No regular representation is implemented here.
    """
    pools = [
        [g for g in G.elements if (2*g[0]+g[1]) % 4 == c]
        for c in range(4)
    ]
    recipes = {
        3: ((0,1,2), (0,0,1), (0,1,3), (0,1,1)),
        4: ((0,1,2,3), (0,0,1,3), (0,1,2,2), (0,0,1,1)),
        5: ((0,0,1,2,3), (0,0,1,1,3),
            (0,1,1,2,2), (0,0,1,1,2)),
    }
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, count in enumerate(row):
            if count == 0:
                out.append([])
                continue
            if count == 1:
                c = (rng.randrange(4)+phase+i+2*j) % 4
                out.append([rng.choice(pools[c])])
                continue

            if count == 2:
                pure = (mode == 1 or
                        (mode == 2 and (i+j+int(side == "B")) % 2 == 0))
                offsets = (0,2) if pure else (0,1)
            else:
                offsets = recipes[count][mode]

            selected = None
            for restart in range(128):
                origin = rng.randrange(4)
                direction = rng.choice((-1,1))
                quota = [(origin+direction*c) % 4 for c in offsets]
                rng.shuffle(quota)
                cell = []
                for c in quota:
                    legal = []
                    for g in pools[c]:
                        if g in cell:
                            continue
                        # Same-coset differences divisible by three have
                        # rotation order <=6, including the involution.
                        if any(g[1] == h[1] and
                               (g[0]-h[0]) % 3 == 0 for h in cell):
                            continue
                        if any(G.mul(G.inv(h), g) == (9,0)
                               for h in cell):
                            continue
                        legal.append(g)
                    if not legal:
                        break
                    cell.append(rng.choice(legal))
                if len(cell) == count:
                    selected = cell
                    break
            if selected is None:
                raise ValueError("Dic9 C4 long-difference recipe exhausted")
            assert len(selected) == len(set(selected)) == count
            out.append(selected)
        grid.append(out)
    return grid


def _d9_kernel_words(cols, rng, trials, memory=()):
    """Retain genuine kernel words across independent information sets.

    XOR shortening preserves the kernel. A high minimum among the
    retained words is a bounded-search reading, never a certificate.
    """
    retained = sorted(
        {w for w in memory if w}, key=lambda w: (w.bit_count(), w))[:32]
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

        words = sorted(set(retained+dependencies),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            reducers = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))

        discoveries = set(words+retained)
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:32]
    return tuple(retained)


def _d9_component_words(cols, fibers, q, rng, trials, memory=()):
    """Search full components and expand central-pair/C9 dependencies.

    Fiber syndromes are sums of VERIFIED component columns. Expanded
    supports are actual kernel words, not distances inferred from a
    separately implemented quotient lift.
    """
    words = set(_d9_kernel_words(cols, rng, trials, memory))
    words.update(_d9_norm_words(cols, fibers, q))
    for h in (2,9):
        supports = [
            fiber << (block*q)
            for block in range(len(cols)//q) for fiber in fibers[h]
        ]
        projected = []
        for support in supports:
            syndrome, remaining = 0, support
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            projected.append(syndrome)

        dependencies = _d9_kernel_words(
            projected, rng, max(2, trials//2))
        for dependency in dependencies:
            expanded = 0
            while dependency:
                bit = dependency & -dependency
                expanded ^= supports[bit.bit_length()-1]
                dependency ^= bit
            if expanded:
                words.add(expanded)
    return tuple(sorted(words, key=lambda w: (w.bit_count(), w))[:48])


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
        rng = random.Random(BASE_SEED*10007 + 36015197 + 100003*slot)

        def component_pool(side, quota):
            counts = sa if side == "A" else sb
            strata, seen = {}, set()

            def key(item):
                load = sum(w.bit_count() <= item["d"]
                           for w in item["words"])
                return (item["floor"], min(item["d"], 24),
                        -load, -item["cycles"], -item["rank"])

            def admit(grid):
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen:
                    return
                seen.add(signature)
                A, B = ((grid, zero_b) if side == "A"
                        else (zero_a, grid))
                hx, hz, _ = lifted_matrices(G, A, B)
                ar, br, ac, bc = _component_data(hx, hz, q)
                rows, cols = (ar, ac) if side == "A" else (br, bc)
                floor = _short_floor(cols)
                if floor < 5:
                    return
                words = _d9_component_words(cols, fibers, q, rng, 3)
                if not words:
                    return
                rank = gf2_rank(rows)
                item = dict(
                    grid=grid, cols=cols, rank=rank, floor=floor,
                    d=words[0].bit_count(), words=words,
                    cycles=_component_cycles(rows))
                bucket = strata.setdefault(rank, [])
                bucket.append(item)
                bucket.sort(key=key, reverse=True)
                del bucket[3:]

            for attempt in range(quota):
                admit(_d9_draw(
                    rng, G, counts, mode, side, slot+attempt))

            positions = [(i,j) for i,row in enumerate(counts)
                         for j,count in enumerate(row) if count]
            for step in range(24):
                pool = sorted(
                    (x for bucket in strata.values() for x in bucket),
                    key=key, reverse=True)
                if not pool:
                    admit(_d9_draw(
                        rng, G, counts, mode, side, slot+quota+step))
                    continue
                parent = rng.choice(pool[:min(5,len(pool))])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                replacement = _d9_draw(
                    rng, G, counts, mode, side, slot+quota+step)
                pressure = [
                    1 + sum(((w >> (j*q)) & mask).bit_count()
                            for w in parent["words"][:8])
                    for i,j in positions
                ]
                # Change actual voltages in witness-heavy blocks; never
                # apply a common translation, automorphism or permutation.
                for _ in range(2 if step % 6 == 5 else 1):
                    i,j = rng.choices(
                        positions, weights=pressure, k=1)[0]
                    grid[i][j] = replacement[i][j]
                admit(grid)

            pool = sorted(
                (x for bucket in strata.values() for x in bucket),
                key=key, reverse=True)[:10]
            for item in pool:
                item["words"] = _d9_component_words(
                    item["cols"], fibers, q, rng, 28, item["words"])
                item["d"] = min(item["d"],
                                item["words"][0].bit_count())
            pool.sort(key=key, reverse=True)
            if not pool:
                return []

            # Keep competitive rank strata for exact joint rank selection.
            # A lower component rank is not itself a promised LP k gain.
            threshold = pool[0]["d"]-2
            selected, ranks = [], set()
            for item in pool:
                if item["d"] < threshold or item["rank"] in ranks:
                    continue
                selected.append(item)
                ranks.add(item["rank"])
                if len(selected) == 6:
                    break
            for item in pool:
                if len(selected) == 6:
                    break
                if all(item is not old for old in selected):
                    selected.append(item)
            return selected

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


# The historical Dic9 bank above is inactive. These policies do not call
# its presentation-specific drawers, fibers or component-word screens.
_C36_B_COUNTS = (
    ((3,3,1,1), (1,1,3,3)),
    ((3,2,2,1), (1,2,2,3)),
    ((2,2,2,2), (2,2,2,2)),
    ((2,3,1,2), (2,1,2,3)),
    ((3,2,2,1), (1,1,2,4)),
)
_C36_A_GAPS = (
    (1,2,4,7,11,17),
    (1,3,5,8,13,17),
    (1,1,5,7,10,13),
    (2,3,7,9,11,17),
)


def _c36_syndrome(cols, word):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= cols[bit.bit_length()-1]
        word ^= bit
    return syndrome


def _c36_draw(local, G, counts, side, mode, lane):
    """Independent voltages within explicit relative-support architectures."""
    from math import gcd
    units = [a for a in range(1, 36) if gcd(a, 36) == 1]
    templates = ((0,1,4), (0,1,7), (0,2,5))
    gaps = list(_C36_A_GAPS[mode])
    if lane:
        # Change one intrinsic gap, rather than scaling the entire design.
        legal = [a for a in range(1, 36)
                 if gcd(gcd(a, gaps[2]), 36) == 1]
        gaps[5] = legal[(legal.index(gaps[5])+3*lane) % len(legal)]

    grid = []
    for i, sizes in enumerate(counts):
        row, used = [], set()
        for j, size in enumerate(sizes):
            if size == 0:
                row.append([])
                continue
            if side == "A":
                anchor = local.randrange(36)
                gap = gaps[3*i+j] * local.choice((-1,1))
                row.append([(anchor,), ((anchor+gap) % 36,)])
                continue

            best = None
            for attempt in range(24):
                anchor = local.randrange(36)
                if size == 1:
                    residues = [anchor]
                elif size == 2:
                    gap = local.choice(units)
                    residues = [anchor, (anchor+gap) % 36]
                elif size == 3 and mode:
                    step = local.choice(units)
                    residues = [(anchor+step*t) % 36
                                for t in templates[mode-1]]
                else:
                    residues = local.sample(range(36), size)
                if len(set(residues)) != size:
                    continue
                if size >= 3 and len({a % 2 for a in residues}) < 2:
                    continue
                differences = {(a-b) % 36 for a in residues
                               for b in residues if a != b}
                cost = (size*(size-1)-len(differences)
                        + len(differences & used))
                if best is None or cost < best[0]:
                    best = (cost, residues, differences)
                if cost == 0:
                    break
            if best is None:
                raise ValueError("cyclic support recipe exhausted")
            row.append([(a,) for a in best[1]])
            used.update(best[2])
        grid.append(row)
    return grid


def _c36_cofactor_words(G, grid, cols, inverse):
    """Commuting-minor constructions, accepted only at zero syndrome.

    This constructs group-algebra words, not a second matrix lift.
    Inverting B labels matches its verified component action.
    """
    from itertools import combinations
    entries = [
        [set(G.inv(g) if inverse else g for g in cell) for cell in row]
        for row in grid]

    def product(left, right):
        out = set()
        for g in left:
            for h in right:
                term = G.mul(g, h)
                if term in out:
                    out.remove(term)
                else:
                    out.add(term)
        return out

    found = set()
    for triple in combinations(range(len(grid[0])), 3):
        minors = {}
        for omitted in triple:
            a, b = [j for j in triple if j != omitted]
            minors[omitted] = (
                product(entries[0][a], entries[1][b])
                ^ product(entries[0][b], entries[1][a]))
        for shift in G.elements:
            word = 0
            for j, minor in minors.items():
                for g in minor:
                    word ^= 1 << (j*G.q+G.idx(G.mul(g, shift)))
            if word and _c36_syndrome(cols, word) == 0:
                found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:48])


def _c36_subgroup_words(cols, partitions, q, local):
    """Actual kernels in listed indicator subspaces; no exclusion claim."""
    found = set()
    for fibers in partitions:
        supports = [fiber << (block*q)
                    for block in range(len(cols)//q) for fiber in fibers]
        projected = [_c36_syndrome(cols, word) for word in supports]
        coefficients = _d9_kernel_words(projected, local, 3)
        for coefficient in coefficients:
            expanded = 0
            while coefficient:
                bit = coefficient & -coefficient
                expanded ^= supports[bit.bit_length()-1]
                coefficient ^= bit
            if expanded:
                assert _c36_syndrome(cols, expanded) == 0
                found.add(expanded)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:48])


def _c36_signature(hx, hz, n):
    """Strengthen the uncoloured invariant with weighted walk profiles.

    Unequal signatures certify nonisomorphism. Equality is conservatively
    rejected and is not claimed to establish graph isomorphism.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    incidents = [[] for _ in range(n)]
    for r, original in enumerate(rows):
        while original:
            bit = original & -original
            incidents[bit.bit_length()-1].append(r)
            original ^= bit
    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1
    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = Counter()
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] += multiplicity*other
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            sum(neighbors.get(t, 0)*value
                for t, value in square.items()),
            tuple(sorted(Counter(square.values()).items()))))
    return (n, tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@__import__("functools").lru_cache(maxsize=1)
def _c36_bank(G):
    """Pure deterministic bank of independently optimized cyclic tasks."""
    assert G.q == 36
    assert A_SHAPE == (2,3) and B_SHAPE == (2,4)
    q = G.q
    mask = (1 << q)-1
    sa = ((2,2,2), (2,2,2))
    zero_a = [[[] for _ in range(3)] for _ in range(2)]
    zero_b = [[[] for _ in range(4)] for _ in range(2)]

    # Compile actions only from scaffold-generated single-term matrices.
    # Linear superposition below does not implement a regular action.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(3 if side == "A" else 4)]
                    for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero_b) if side == "A" else (zero_a, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)

    def component(side, grid):
        cols = [0]*(len(grid[0])*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for t, syndrome in enumerate(block):
                        cols[j*q+t] ^= syndrome << (i*q)
        return cols

    partitions = []
    for size in (2,3,4,6,9,12,18):
        unseen, fibers = set(range(q)), []
        while unseen:
            anchor = G.elements[min(unseen)]
            orbit = {G.idx(G.mul(anchor, (t*(q//size),)))
                     for t in range(size)}
            assert len(orbit) == size
            unseen.difference_update(orbit)
            fibers.append(sum(1 << t for t in orbit))
        partitions.append(tuple(sorted(fibers)))

    def pool(local, side, counts, mode, lane, batch):
        strata, seen = {}, set()

        def key(item):
            load = sum(w.bit_count() <= item["d"]+1
                       for w in item["words"])
            return (item["d"], -load, -item["cycles"])

        def admit(grid):
            labels = tuple(tuple(sorted(cell))
                           for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            cols = component(side, grid)
            if _short_floor(cols) < 5:
                return
            seeds = _c36_cofactor_words(G, grid, cols, side == "B")
            words = _d9_kernel_words(cols, local, 3, seeds)
            if not words:
                return
            rank = gf2_rank(cols)
            # Columns are the transpose representation of this component.
            rows = _columns(cols, 2*q)
            item = dict(
                grid=grid, cols=cols, words=words,
                d=words[0].bit_count(), rank=rank,
                cycles=_component_cycles(rows))
            bucket = strata.setdefault(rank, [])
            bucket.append(item)
            bucket.sort(key=key, reverse=True)
            del bucket[2:]

        quota = (64 if side == "A" else 96)+24*batch
        for attempt in range(quota):
            admit(_c36_draw(local, G, counts, side, mode, lane))

        # Repairs replace actual cells, not an automorphic image.
        positions = [(i,j) for i,row in enumerate(counts)
                     for j,size in enumerate(row) if size]
        for step in range(16):
            candidates = sorted(
                (item for bucket in strata.values() for item in bucket),
                key=key, reverse=True)
            if not candidates:
                break
            parent = local.choice(candidates[:min(4, len(candidates))])
            grid = [[list(cell) for cell in row] for row in parent["grid"]]
            replacement = _c36_draw(local, G, counts, side, mode, lane)
            pressure = [
                1+sum(((word >> (j*q)) & mask).bit_count()
                      for word in parent["words"][:8])
                for i,j in positions]
            for _ in range(2 if step % 6 == 5 else 1):
                i,j = local.choices(positions, weights=pressure, k=1)[0]
                grid[i][j] = replacement[i][j]
            admit(grid)

        candidates = sorted(
            (item for bucket in strata.values() for item in bucket),
            key=key, reverse=True)[:10]
        for item in candidates:
            seeds = tuple(item["words"])+_c36_subgroup_words(
                item["cols"], partitions, q, local)
            words = _d9_kernel_words(item["cols"], local, 24, seeds)
            item["words"] = words
            item["d"] = min(item["d"], words[0].bit_count())
        candidates.sort(key=key, reverse=True)
        finalists = candidates[:5]
        for item in finalists:
            words = _d9_kernel_words(
                item["cols"], local, 48, item["words"])
            item["words"] = words
            item["d"] = min(item["d"], words[0].bit_count())
        return sorted(finalists, key=key, reverse=True)[:4]

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("dx", item["n"]+1),
                item.get("dz", item["n"]+1))
        load = sum(w.bit_count() <= d+1
                   for side in ("a", "b")
                   for w in item[side]["words"])
        return (item["k"]*d*d/item["n"], d, -load,
                -item["a"]["cycles"]-item["b"]["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        sb = _C36_B_COUNTS[architecture]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        local = random.Random(BASE_SEED*100003+36015317+104729*slot)
        chosen = None

        for batch in range(3):
            ap = pool(local, "A", sa, mode, slot//20, batch)
            bp = pool(local, "B", sb, mode, slot//20, batch)
            assembled = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (125*n <= 1000*k < 175*n):
                        continue
                    ar, br, ac, bc = _component_data(hx, hz, q)
                    assert ac == a["cols"] and bc == b["cols"]
                    assembled.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            assembled.sort(key=merit, reverse=True)
            finalists = []
            for item in assembled[:10]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _d9_low4(hx, hz, n) or _d9_low4(hz, hx, n):
                    continue
                a_seeds = [
                    sum(((word >> (j*q)) & mask)
                        << ((4*j+beta)*q) for j in range(3))
                    for word in item["a"]["words"] for beta in range(4)]
                b_seeds = [
                    word << (4*j*q)
                    for word in item["b"]["words"] for j in range(3)]
                dx, wx = _d9_logical_probe(hz, hx, n, local, 6, b_seeds)
                dz, wz = _d9_logical_probe(hx, hz, n, local, 6, a_seeds)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx, dz=dz, wx=wx, wz=wz,
                            a_seeds=a_seeds, b_seeds=b_seeds)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # Equal independent budgets for EVERY finalist; retain all
            # earlier lighter witnesses so readings can only decrease.
            for item in finalists:
                dx, wx = _d9_logical_probe(
                    item["hz"], item["hx"], item["n"], local, 20,
                    tuple(item["wx"])+tuple(item["b_seeds"]))
                dz, wz = _d9_logical_probe(
                    item["hx"], item["hz"], item["n"], local, 20,
                    tuple(item["wz"])+tuple(item["a_seeds"]))
                item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)

            for item in sorted(finalists, key=merit, reverse=True):
                signature = _c36_signature(
                    item["hx"], item["hz"], item["n"])
                if signature in emitted:
                    continue
                A, B = item["A"], item["B"]
                assert item["n"] == 576 and item["k"] >= 72
                assert css_ok(item["hx"], item["hz"])
                assert max(row.bit_count()
                           for row in item["hx"]+item["hz"]) <= W_MAX
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (A, B) for row in grid for cell in row)
                emitted.add(signature)
                chosen = dict(
                    cells={
                        "A": tuple(tuple(tuple(cell) for cell in row)
                                   for row in A),
                        "B": tuple(tuple(tuple(cell) for cell in row)
                                   for row in B)},
                    k=item["k"], card=(architecture, mode),
                    selection=merit(item))
                break
            if chosen is not None:
                break

        # Never replace a failed task with an emitted orbit copy.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the independently optimized cyclic task, without relabeling."""
    item = _c36_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh cyclic filling passed target-rate screens")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the full jointly selected support, without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting cell relatives and component dependency screens through six eliminate sector-1 weight-twelve logicals on GL(2,3), (1,2)x(2,5); all five measured parent codes retained both X and Z weight-twelve sector-1 witnesses.
# REFUTED: Full-group-generating A relatives and subgroup screening raise GL(2,3) five-code fitness above 38.25; the current measurements instead scored 37.7542, with four d30s=12 readings, one mixed-sector d30s=11 reading, and four occupancy-recipe failures.
# REFUTED: Dic9 order-nine same-coset binomial cards improve on the long-rotation/mixed cards; every parent mode-three slot had d2s=9, whereas the five measured leaders from other modes retained d30s=16.
# REFUTED: C4-aware labeling and central-pair component screening eliminate sub-sixteen witnesses on every Dic9 support card; current slots 16,17,19 instead returned d2s=12,12,9.
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
