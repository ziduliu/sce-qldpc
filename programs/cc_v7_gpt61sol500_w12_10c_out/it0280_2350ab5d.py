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
CONCEPT = ("C35, order 35, A=(2,3), B=(2,4): independently selecting components against commuting-cofactor cancellations and C5/C7-invariant kernel words should exceed the 0.150 frontier fitness 43.2; measured evaluation can refute this.")
GROUP = ("Jump from Dic9 to the genuinely different cyclic C35. C37 reached 43.2 on these shapes, and cyclic C43/C46 performed well on equal two-check beds. C35's bed is absent from BEDS; odd order and n=560 motivate it, not a distance-transfer claim. Its order-five/seven subgroups are explicit possible sources of light words and are screened.")
SHAPES = ("Keep (2,3)x(2,4) for cell 0.150: n=560, kb=70, guaranteed rate 1/8, maximum admissible lift order 37, and component variables/checks 105/70 and 140/70. Parent limiting weight-sixteen X families occupied sector-one Acol/Bcol blocks across every column, with no reported minimum in Arow/Brow. Retain exact assembled 0.125<=k/n<0.175.")
FILLING = ("Twenty policies cross five B count geometries with four intrinsic difference rules. A cells are doubletons, B row sums are eight and column sums at most five: HX<=11 and HZ<=12. Compile actions only from verified unit lifts. Retain syndrome-verified commuting cofactors and expanded C5/C7 kernel witnesses, independently optimize components with witness-directed repairs and rank diversity, and rescreen every joint finalist with exact stabilizer membership and logical-four exclusion. Unequal uncoloured incidence invariants certify emitted nonisomorphism. Bounded witness readings remain upper estimates, not distance certificates.")
FAMILY_NAME = "r015_C35_23x24_cofactor_subgroup_rankrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    # C35, encoded additively; not the parent's dicyclic presentation.
    if not valid_q(q):
        raise ValueError("this cyclic bed requires order 35")
    els = [(a,) for a in range(35)]

    def red(g):
        return (g[0] % 35,)

    def mul(g, h):
        return ((g[0] + h[0]) % 35,)

    def inv(g):
        return ((-g[0]) % 35,)

    def idx(g):
        return g[0] % 35

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


# Twenty architecture/quotient policies, not relabelings of one filling.
# Listed maxima (HX,HZ): (10,12), (10,12), (10,12), (11,12), (11,12).
# The fifth B profile trades one degree-four column for degrees five/three.
_D9_SUPPORTS = (
    (((0,3,3), (4,1,1)), ((1,2,2,3), (3,2,2,1))),
    (((2,2,2), (2,2,2)), ((3,3,1,1), (1,1,3,3))),
    (((1,2,3), (3,2,1)), ((0,2,3,3), (4,2,1,1))),
    (((3,2,2), (1,2,2)), ((3,2,2,1), (1,2,2,3))),
    (((2,2,2), (2,2,2)), ((1,2,2,3), (4,2,2,0))),
)


def _d9_draw(rng, G, counts, mode, side, phase):
    """C4-resolved cells; group operations constrain labels, not a lift."""
    pools = [[g for g in G.elements if (2*g[0]+g[1]) % 4 == c]
             for c in range(4)]
    recipes = {
        3: ((1,1,1,0), (2,1,0,0), (2,0,0,1), (1,1,0,1)),
        4: ((1,1,1,1), (2,1,1,0), (2,2,0,0), (2,0,1,1)),
        5: ((2,1,1,1), (2,2,1,0), (3,1,0,1), (2,1,0,2)),
    }

    def ratios(cell):
        if side == "A":
            return {G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h}
        return {G.mul(G.inv(h), g)
                for g in cell for h in cell if g != h}

    grid = []
    for i, row in enumerate(counts):
        out, used = [], set()
        for j, count in enumerate(row):
            if count <= 1:
                out.append([] if not count else [G.pick(rng)])
                continue

            if count == 2:
                long_pair = (
                    mode == 1 and (i+j+int(side == "B")) % 2 == 0
                    or mode == 3 and side == "B" and j % 2 == 0)
                # chi difference two gives a rotation; odd differences
                # give order-four outer elements. Residue separation
                # below makes the rotation's order exactly eighteen.
                quota = (1,0,1,0) if long_pair else (1,1,0,0)
            else:
                quota = recipes[count][mode]

            best, accepted = None, 0
            for attempt in range(160):
                shift = rng.randrange(4)
                direction = rng.choice((-1, 1))
                cell = []
                for c, amount in enumerate(quota):
                    cell.extend(rng.sample(
                        pools[(shift+direction*c) % 4], amount))

                if any(g[1] == h[1] and (g[0]-h[0]) % 3 == 0
                       for t, g in enumerate(cell) for h in cell[:t]):
                    continue
                if any(G.mul(G.inv(h), g) == (9,0)
                       for t, g in enumerate(cell) for h in cell[:t]):
                    continue

                relative = ratios(cell)
                cost = (
                    count*(count-1)-len(relative)+len(relative & used),
                    -len(relative))
                if best is None or cost < best[0]:
                    best = (cost, cell, relative)
                accepted += 1
                if cost[0] == 0 or accepted >= 8:
                    break

            if best is None:
                raise ValueError("Dic9 C4/residue recipe exhausted")
            assert len(best[1]) == len(set(best[1])) == count
            out.append(best[1])
            used.update(best[2])
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
    """Actual kernels among bounded unions of equal-weight coset indicators.

    Different listed partitions may overlap. Coefficient-weight cutoffs
    are conservative restrictions, not claims of exhaustive decoding.
    Every expanded witness has zero verified component syndrome.
    """
    from itertools import combinations
    found = set()
    for h in ((2,3,4,6,9,12,18) if deep else (3,6,9,18)):
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
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:128])


def _d9_kernel_words(cols, rng, trials, seeds=()):
    """Retain genuine component kernels across information sets.

    All moves XOR verified kernel supports. Shortening and recombination
    do not certify the absence of lighter words.
    """
    retained = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:48]
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

        words = sorted(set(dependencies+retained),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(retained)
        discoveries.update(words[:48])
        for sweep in range(2):
            reducers = words[:40]
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
            discoveries.update(words[:48])

        short = sorted(discoveries, key=lambda w: (w.bit_count(), w))[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        if trial % 4 == 0:
            for i, word in enumerate(short[:10]):
                for j in range(i):
                    pair = word ^ short[j]
                    for h in range(j):
                        candidate = pair ^ short[h]
                        if candidate:
                            discoveries.add(candidate)

        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:48]
        if retained and retained[0].bit_count() <= 4:
            break
    return tuple(retained)


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

    order = list(range(n))
    memory = sorted(set(seeds), key=int.bit_count)[:48]
    for trial in range(trials):
        rng.shuffle(order)
        # Alternate information sets that expose either LP sector first.
        if trial % 3 != 2:
            factor = A_SHAPE[1]*B_SHAPE[1] + A_SHAPE[0]*B_SHAPE[0]
            split = A_SHAPE[1]*B_SHAPE[1]*(n//factor)
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

    # Add every outer C4 and Dic3 subgroup, with both coset orientations.
    # This constructs supports only; syndromes still use verified columns.
    for h in (4, 12):
        subgroups = set()
        for a in range(18):
            generators = ((18//(h//2), 0), (a, 1))
            members, seen = [G.e], {G.e}
            for g in members:
                for generator in generators:
                    value = G.mul(g, generator)
                    if value not in seen:
                        seen.add(value)
                        members.append(value)
            assert len(members) == h
            subgroups.add(tuple(sorted(G.idx(g) for g in members)))

        indicators = set()
        for subgroup in subgroups:
            for orientation in range(2):
                unseen = set(range(q))
                while unseen:
                    g = G.elements[min(unseen)]
                    coset = {
                        G.idx(G.mul(g, G.elements[t]) if orientation == 0
                              else G.mul(G.elements[t], g))
                        for t in subgroup}
                    assert len(coset) == h
                    unseen.difference_update(coset)
                    indicators.add(sum(1 << t for t in coset))
        fibers[h] = tuple(sorted(indicators))

    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        sa, sb = _D9_SUPPORTS[geometry]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED*10007 + 36015191 + 100003*slot)

        def component_pool(side, quota):
            counts = sa if side == "A" else sb
            strata, seen = {}, set()

            def key(item):
                d = min(item["d"], 24)
                load = sum(w.bit_count() <= d+1 for w in item["words"])
                return (item["floor"], d, -load, -item["cycles"])

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
                rank = gf2_rank(rows)
                structured = _d9_norm_words(cols, fibers, q)
                words = _d9_kernel_words(cols, rng, 4, structured)
                if not words:
                    return
                item = dict(
                    grid=grid, cols=cols, rank=rank, floor=floor,
                    d=words[0].bit_count(), words=words,
                    cycles=_component_cycles(rows))
                bucket = strata.setdefault(rank, [])
                bucket.append(item)
                bucket.sort(key=key, reverse=True)
                del bucket[2:]

            for attempt in range(quota):
                admit(_d9_draw(
                    rng, G, counts, mode, side, slot+attempt))

            # Each repair changes one or two actual cells. Choose a rank
            # stratum first so that a high-rank tie cannot erase diversity.
            positions = [(i, j) for i, row in enumerate(counts)
                         for j, count in enumerate(row) if count]
            repairs = 24 if side == "A" else 48
            for step in range(repairs):
                if not strata:
                    admit(_d9_draw(
                        rng, G, counts, mode, side, slot+quota+step))
                    continue
                rank = rng.choice(sorted(strata))
                parent = rng.choice(strata[rank])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                proposal = _d9_draw(
                    rng, G, counts, mode, side, slot+quota+step)
                pressure = [
                    1 + sum(((word >> (j*q)) & mask).bit_count()
                            for word in parent["words"][:12])
                    for i, j in positions]
                for move in range(2 if step % 7 == 6 else 1):
                    i, j = rng.choices(
                        positions, weights=pressure, k=1)[0]
                    grid[i][j] = proposal[i][j]
                admit(grid)

            # First retain one candidate per rank, then fill spare places.
            # These strata are measured component ranks, not rate promises.
            first = sorted((bucket[0] for bucket in strata.values()),
                           key=key, reverse=True)
            pool = first[:8]
            if len(pool) < 8:
                second = sorted(
                    (bucket[1] for bucket in strata.values()
                     if len(bucket) > 1), key=key, reverse=True)
                pool.extend(second[:8-len(pool)])

            for item in pool:
                structured = _d9_norm_words(
                    item["cols"], fibers, q, deep=True)
                words = _d9_kernel_words(
                    item["cols"], rng, 32 if side == "A" else 48,
                    tuple(item["words"])+structured)
                item["words"] = words
                item["d"] = min(item["d"], words[0].bit_count())

            # Again preserve rank strata before spending spare capacity
            # on a second candidate from an already represented rank.
            pool.sort(key=key, reverse=True)
            selected, ranks = [], set()
            for item in pool:
                if item["rank"] not in ranks:
                    selected.append(item)
                    ranks.add(item["rank"])
                    if len(selected) == 6:
                        break
            for item in pool:
                if len(selected) == 6:
                    break
                if not any(item is old for old in selected):
                    selected.append(item)
            return selected

        winner = None
        for retry in range(2):
            ap = component_pool("A", 80+32*retry+4*(slot//20))
            bp = component_pool("B", 144+48*retry+8*(slot//20))
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
            for item in proposals[:12]:
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
                load = sum(
                    word.bit_count() <= d+1
                    for component in (item["ai"], item["bi"])
                    for word in component["words"])
                return (item["k"]*d*d/item["n"], d,
                        min(item["dx"], item["dz"]), -load,
                        -item["cycles"])

            finalists.sort(key=merit, reverse=True)
            # Every finalist gets the same independent budget. An
            # undeepened candidate cannot win on an optimistic reading.
            for item in finalists:
                dx, wx = _d9_logical_probe(
                    item["hz"], item["hx"], item["n"], rng, 20,
                    tuple(item["wx"])+tuple(
                        word << (j*cb*q)
                        for word in item["bi"]["words"]
                        for j in range(ca)))
                dz, wz = _d9_logical_probe(
                    item["hx"], item["hz"], item["n"], rng, 20,
                    tuple(item["wz"])+tuple(
                        sum(((word >> (j*q)) & mask)
                            << ((j*cb+beta)*q) for j in range(ca))
                        for word in item["ai"]["words"]
                        for beta in range(cb)))
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                item["wx"], item["wz"] = wx, wz
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


# The historical Dic9 builders above are inactive. Only the C35 bank
# below supplies NEW_CTX/CELL.
_C35_SUPPORTS = (
    ((3,3,1,1), (1,1,3,3)),
    ((3,2,2,1), (1,2,2,3)),
    ((2,3,1,2), (2,1,2,3)),
    ((3,2,2,1), (1,1,2,4)),
    ((2,2,2,2), (2,2,2,2)),
)


def _c35_draw(local, G, counts, mode, side):
    """Intrinsic difference policies, with fresh independent voltages."""
    from math import gcd
    q = G.q
    units = [d for d in range(1,q) if gcd(d,q) == 1]
    order7 = [d for d in range(1,q) if gcd(d,q) == 5]
    order5 = [d for d in range(1,q) if gcd(d,q) == 7]
    used_gaps = set()
    grid = []
    repeated = local.choice(units)
    for i, sizes in enumerate(counts):
        row = []
        for j, size in enumerate(sizes):
            if not size:
                row.append([])
                continue
            if size == 1:
                row.append([G.pick(local)])
                continue
            if size == 2:
                pool = units
                if side == "A" and mode == 3:
                    if (i,j) == (0,0):
                        pool = order7
                    elif (i,j) == (1,1):
                        pool = order5
                if side == "B" and mode == 2 and i == 0 and j == 1:
                    pool = order5
                if side == "A" and mode == 2 and i == 0 and j < 2:
                    delta = repeated
                else:
                    legal = [d for d in pool
                             if mode != 1 or min(d,q-d) not in used_gaps]
                    delta = local.choice(legal or pool)
                used_gaps.add(min(delta,q-delta))
                anchor = local.randrange(q)
                row.append([(anchor,), ((anchor+delta) % q,)])
                continue

            selected = None
            for attempt in range(2048):
                residues = local.sample(range(q), size)
                gaps = [
                    (residues[a]-residues[b]) % q
                    for a in range(size) for b in range(a)]
                unsigned = [min(d,q-d) for d in gaps]
                # Sidon-like internal packing is intrinsic to the support.
                if len(unsigned) != len(set(unsigned)):
                    continue
                f5 = sum(d % 5 == 0 for d in gaps)
                f7 = sum(d % 7 == 0 for d in gaps)
                if mode == 0 and (f5 or f7):
                    continue
                if mode == 1 and not (f5 == 1 and f7 == 0):
                    continue
                if mode == 2 and not (f7 == 1 and f5 == 0):
                    continue
                selected = [(a,) for a in residues]
                break
            if selected is None:
                raise ValueError("C35 intrinsic difference policy exhausted")
            assert len(selected) == len(set(selected)) == size
            row.append(selected)
        grid.append(row)
    return grid


def _c35_cofactors(G, grid, cols, inverse):
    """Actual commuting-minor words, accepted only at zero syndrome."""
    from itertools import combinations
    entries = [[set(G.inv(g) if inverse else g for g in cell)
                for cell in row] for row in grid]

    def product(left, right):
        result = set()
        for g in left:
            for h in right:
                term = G.mul(g,h)
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
        return result

    found = set()
    for triple in combinations(range(len(grid[0])), 3):
        minors = {}
        for omitted in triple:
            a,b = [j for j in triple if j != omitted]
            minors[omitted] = (
                product(entries[0][a], entries[1][b])
                ^ product(entries[0][b], entries[1][a]))
        for shift in G.elements:
            word = 0
            for j, minor in minors.items():
                for g in minor:
                    word ^= 1 << (j*G.q + G.idx(G.mul(g,shift)))
            if not word:
                continue
            remaining, syndrome = word, 0
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            if syndrome == 0:
                found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(),w)))


def _c35_subgroup_words(cols, partitions, q, local, trials):
    """Expand genuine restricted dependencies using verified syndromes."""
    found = set()
    for fibers in partitions:
        supports = [fiber << (j*q)
                    for j in range(len(cols)//q) for fiber in fibers]
        projected = []
        for support in supports:
            remaining, syndrome = support, 0
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            projected.append(syndrome)
        dependencies = _d9_kernel_words(projected, local, trials)
        for dependency in dependencies:
            expanded = 0
            while dependency:
                bit = dependency & -dependency
                expanded ^= supports[bit.bit_length()-1]
                dependency ^= bit
            if expanded:
                found.add(expanded)
    return tuple(sorted(found, key=lambda w: (w.bit_count(),w))[:48])


def _c35_fingerprint(hx, hz, n):
    """Uncoloured incidence invariant, refined by overlap-graph triangles.

    Inequality certifies nonisomorphism; equality is conservatively
    rejected and is never asserted to prove isomorphism.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            if row & rows[j]:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = Counter()
    for i, neighbors in enumerate(adjacency):
        remaining, triangles = neighbors, 0
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        profiles[(rows[i].bit_count(), neighbors.bit_count(),
                  triangles//2)] += 1
    return (_d9_incidence_signature(hx,hz,n),
            tuple(sorted(profiles.items())))


@__import__("functools").lru_cache(maxsize=1)
def _c35_bank(G):
    """Pure deterministic bank of separately optimized structural tasks."""
    assert G.q == 35 and A_SHAPE == (2,3) and B_SHAPE == (2,4)
    q = G.q
    mask = (1 << q)-1
    zero_a = [[[] for _ in range(3)] for _ in range(2)]
    zero_b = [[[] for _ in range(4)] for _ in range(2)]

    # Compile regular actions exclusively from scaffold-generated units.
    # Subsequent compilation is F2-linear superposition, not another lift.
    atlas = {}
    for side in ("A","B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(3 if side == "A" else 4)]
                    for _ in range(2)]
            unit[0][0] = [g]
            A,B = (unit,zero_b) if side == "A" else (zero_a,unit)
            hx,hz,_ = lifted_matrices(G,A,B)
            ar,br,ac,bc = _component_data(hx,hz,q)
            blocks.append(tuple((ac if side == "A" else bc)[:q]))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*(len(grid[0])*q)
        for i,row in enumerate(grid):
            for j,cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for t,syndrome in enumerate(block):
                        cols[j*q+t] ^= syndrome << (i*q)
        return cols

    partitions = []
    for order in (5,7):
        unseen, fibers = set(range(q)), []
        while unseen:
            g = G.elements[min(unseen)]
            orbit = {G.idx(G.mul(g,(t*(q//order),)))
                     for t in range(order)}
            assert len(orbit) == order
            unseen.difference_update(orbit)
            fibers.append(sum(1 << t for t in orbit))
        partitions.append(tuple(sorted(fibers)))

    def component_pool(local, side, counts, mode, batch):
        strata, seen = {}, set()
        quota = (80 if side == "A" else 128) + 32*batch

        def merit(item):
            d = min(item["d"],24)
            load = sum(w.bit_count() <= d+1 for w in item["words"])
            return (d,-load,-item["cycles"])

        def admit(grid):
            labeling = tuple(tuple(sorted(cell))
                             for row in grid for cell in row)
            if labeling in seen:
                return
            seen.add(labeling)
            cols = compile_component(side,grid)
            if _short_floor(cols) < 5:
                return
            cofactors = _c35_cofactors(G,grid,cols,side == "B")
            restricted = _c35_subgroup_words(
                cols,partitions,q,local,2)
            words = _d9_kernel_words(
                cols,local,3 if side == "A" else 4,
                cofactors+restricted)
            if not words:
                return
            item = dict(
                grid=grid,cols=cols,words=words,
                d=words[0].bit_count(),rank=gf2_rank(cols),
                cycles=_component_cycles(cols))
            bucket = strata.setdefault(item["rank"],[])
            bucket.append(item)
            bucket.sort(key=merit,reverse=True)
            del bucket[2:]

        for attempt in range(quota):
            admit(_c35_draw(local,G,counts,mode,side))

        positions = [(i,j) for i,row in enumerate(counts)
                     for j,size in enumerate(row) if size]
        for step in range(24 if side == "A" else 48):
            if not strata:
                admit(_c35_draw(local,G,counts,mode,side))
                continue
            rank = local.choice(sorted(strata))
            parent = local.choice(strata[rank])
            grid = [[list(cell) for cell in row] for row in parent["grid"]]
            replacement = _c35_draw(local,G,counts,mode,side)
            pressure = [
                1+sum(((word >> (j*q)) & mask).bit_count()
                      for word in parent["words"][:12])
                for i,j in positions]
            # Repair complete columns to preserve the A gap policy.
            # B repairs change actual supports, not a common cell gauge.
            for move in range(2 if step % 7 == 6 else 1):
                i,j = local.choices(positions,weights=pressure,k=1)[0]
                if side == "A":
                    for r in range(2):
                        grid[r][j] = replacement[r][j]
                else:
                    grid[i][j] = replacement[i][j]
            admit(grid)

        first = sorted((bucket[0] for bucket in strata.values()),
                       key=merit,reverse=True)
        pool = first[:8]
        if len(pool) < 8:
            second = sorted(
                (bucket[1] for bucket in strata.values() if len(bucket)>1),
                key=merit,reverse=True)
            pool.extend(second[:8-len(pool)])
        for item in pool:
            restricted = _c35_subgroup_words(
                item["cols"],partitions,q,local,12)
            words = _d9_kernel_words(
                item["cols"],local,24 if side == "A" else 48,
                tuple(item["words"])+restricted)
            item["words"] = words
            item["d"] = min(item["d"],words[0].bit_count())
        pool.sort(key=merit,reverse=True)
        selected, ranks = [], set()
        for item in pool:
            if item["rank"] not in ranks:
                selected.append(item)
                ranks.add(item["rank"])
                if len(selected) == 4:
                    break
        for item in pool:
            if len(selected) == 4:
                break
            if not any(item is old for old in selected):
                selected.append(item)
        # Every retained component receives the same final budget.
        for item in selected:
            words = _d9_kernel_words(
                item["cols"],local,48 if side == "A" else 80,
                item["words"])
            item["words"] = words
            item["d"] = min(item["d"],words[0].bit_count())
        return sorted(selected,key=merit,reverse=True)

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        geometry,mode = divmod(slot % 20,4)
        sa = ((2,2,2),(2,2,2))
        sb = _C35_SUPPORTS[geometry]
        assert _grid_weights(sa,sb) <= W_MAX
        assert _no_dead_lines(sa,sb)
        local = random.Random(
            BASE_SEED*100003 + 35015317 + 104729*slot)
        chosen = None

        def joint_merit(item):
            d = min(item["a"]["d"],item["b"]["d"],
                    item.get("dx",item["n"]+1),
                    item.get("dz",item["n"]+1),24)
            return (item["k"]*d*d/item["n"],d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        for batch in range(3):
            ap = component_pool(local,"A",sa,mode,batch)
            bp = component_pool(local,"B",sb,mode,batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A,B = a["grid"],b["grid"]
                    hx,hz,n = lifted_matrices(G,A,B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (125*n <= 1000*k < 175*n):
                        continue
                    ar,br,ac,bc = _component_data(hx,hz,q)
                    assert ac == a["cols"] and bc == b["cols"]
                    proposals.append(dict(
                        A=A,B=B,hx=hx,hz=hz,n=n,k=k,a=a,b=b))
            proposals.sort(key=joint_merit,reverse=True)

            finalists = []
            for item in proposals[:10]:
                hx,hz,n = item["hx"],item["hz"],item["n"]
                if _d9_low4(hx,hz,n) or _d9_low4(hz,hx,n):
                    continue
                fingerprint = _c35_fingerprint(hx,hz,n)
                if fingerprint in emitted:
                    continue
                a_seeds = tuple(
                    sum(((word >> (j*q)) & mask)
                        << ((j*4+beta)*q) for j in range(3))
                    for word in item["a"]["words"] for beta in range(4))
                b_seeds = tuple(
                    word << (j*4*q)
                    for word in item["b"]["words"] for j in range(3))
                # Validate every supplied full-code seed before it can
                # enter the shortening memory of the generic probe.
                assert all(not any((w & row).bit_count() & 1
                                   for row in hx) for w in a_seeds)
                assert all(not any((w & row).bit_count() & 1
                                   for row in hz) for w in b_seeds)
                dx,wx = _d9_logical_probe(
                    hz,hx,n,local,8,b_seeds)
                dz,wz = _d9_logical_probe(
                    hx,hz,n,local,8,a_seeds)
                if dx > n or dz > n:
                    continue
                item.update(
                    dx=dx,dz=dz,wx=wx,wz=wz,
                    a_seeds=a_seeds,b_seeds=b_seeds,
                    fingerprint=fingerprint)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # Equal independent full-code budgets for every finalist.
            for item in finalists:
                dx,wx = _d9_logical_probe(
                    item["hz"],item["hx"],item["n"],local,24,
                    tuple(item["wx"])+item["b_seeds"])
                dz,wz = _d9_logical_probe(
                    item["hx"],item["hz"],item["n"],local,24,
                    tuple(item["wz"])+item["a_seeds"])
                item["dx"] = min(item["dx"],dx)
                item["dz"] = min(item["dz"],dz)
            if not finalists:
                continue
            winner = max(finalists,key=joint_merit)
            hx,hz,n = winner["hx"],winner["hz"],winner["n"]
            assert n == 560 and winner["k"] >= 70
            assert css_ok(hx,hz)
            assert max(row.bit_count() for row in hx+hz) <= W_MAX
            assert all(len(cell) == len(set(cell)) <= 5
                       for grid in (winner["A"],winner["B"])
                       for row in grid for cell in row)
            emitted.add(winner["fingerprint"])
            chosen = dict(
                cells={
                    side: tuple(tuple(tuple(cell) for cell in row)
                                for row in winner[side])
                    for side in ("A","B")},
                k=winner["k"],card=(geometry,mode),
                selection=joint_merit(winner))
            break
        # A failed structural task is never replaced by an emitted copy.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this C35 policy's fresh, independently screened design."""
    item = _c35_bank(G)[idx]
    if item is None:
        raise ValueError("C35 policy found no fresh screened 0.150 filling")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the complete immutable selection, without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting cell relatives and component dependency screens through six eliminate sector-1 weight-twelve logicals on GL(2,3), (1,2)x(2,5); all five measured parent codes retained both X and Z weight-twelve sector-1 witnesses.
# REFUTED: Full-group-generating A relatives and subgroup screening raise GL(2,3) five-code fitness above 38.25; the current measurements instead scored 37.7542, with four d30s=12 readings, one mixed-sector d30s=11 reading, and four occupancy-recipe failures.
# REFUTED: Dic9's four original occupancy modes are uniformly competitive on (2,3)x(2,4); all fifteen mode-three slots read d2s=9, while the five measured mode-zero/two leaders retained d30s=16.
# REFUTED: C4-resolved labels and B-weighted repairs eliminate the limiting sector-1 weight-sixteen family on Dic9 (2,3)x(2,4); all five measured current-parent leaders retained d30s=16 and sector-1 X minima.
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
