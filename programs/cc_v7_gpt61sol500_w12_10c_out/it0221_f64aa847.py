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
CONCEPT = ("Binary octahedral 2O, order 48, shapes (1,2)x(2,5): full-group-generating A cells and subgroup-coset packing should raise five-code fitness above 43.482 by suppressing the parent's weight-13 sector-one witnesses; this is a falsifiable measured-fitness hypothesis.")
GROUP = ("Jump from Dic12 to the determinant-parity central twist of GL(2,3), constructing binary octahedral 2O. It has a unique involution unlike GL(2,3), and a nonabelian normal Q8 subgroup unlike Dic12's cyclic normal subgroup. The supplied GL(2,3) and Dic12 beds reached 39.0 and 43.5; the different extension is an untried hypothesis, not a transferable distance claim.")
SHAPES = ("Keep (1,2)x(2,5) for cell 0.250: n=576, kb=144, guaranteed rate 1/4, and component checks/variables 48/96 and 96/240. The size window allows q<=49 and this group fixes q=48. All five measured parent X minima had weight 13 entirely in sector one, broadly touching both A columns and B columns; Z minima had weight 14. Verified full ranks retain [0.225,0.275).")
FILLING = ("Five legal count profiles cross packing against C2, C4, C3 and Q8 cosets. Require each A multiplet's anchor relatives to generate all 48 elements; exclude the unique involution difference and prescribe relative orders six/eight in B binomials. Both row-weight sums remain <=12. Preserve verified coset-fiber searches, retained kernel words, witness-directed repairs, stabilizer-tested embeddings and full-matrix probes. Deepen A's restricted search through weight fourteen and penalize retained witnesses through fourteen. Restricted exclusions and randomized readings never certify distance.")
FAMILY_NAME = "r025_binaryOct48_12x25_fullgen_coset_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Binary octahedral 2O: twist GL(2,F3) by its determinant parity.
    # g*h = (-I)^(epsilon(g)*epsilon(h)) times the ordinary product.
    # Determinant parity is unchanged by the central sign, so the
    # sign is a central 2-cocycle and associativity follows.
    if not valid_q(q):
        raise ValueError("binary octahedral 2O requires order 48")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        sign = (2 if (a*d-b*c) % 3 == 2
                and (e*v-f*u) % 3 == 2 else 1)
        return ((sign*(a*e+b*u)) % 3,
                (sign*(a*f+b*v)) % 3,
                (sign*(c*e+d*u)) % 3,
                (sign*(c*f+d*v)) % 3)

    def inv(g):
        a,b,c,d = red(g)
        # For determinant -1 the cocycle cancels the ordinary
        # inverse's determinant factor; the adjugate works in both cases.
        return (d % 3, (-b) % 3, (-c) % 3, a % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

# Each card is (A cell sizes, B column sizes).
# First three: HX<=12, HZ<=12. Fourth: HX<=11, HZ<=12.
# Fifth: HX<=10, HZ<=12. Every row and column is nonzero.
_DIC_SUPPORTS = (
    ((4,4), ((2,1), (2,1), (1,2), (1,2), (2,2))),
    ((4,4), ((3,0), (0,3), (2,1), (1,2), (2,2))),
    ((4,4), ((3,1), (1,3), (1,2), (2,1), (1,1))),
    ((3,5), ((2,1), (2,1), (1,2), (1,2), (1,1))),
    ((3,3), ((2,2), (2,2), (2,2), (2,1), (1,2))),
)


def _oct_parity(g):
    """Determinant -1 is the nontrivial C2 quotient class."""
    return int((g[0]*g[3] - g[1]*g[2]) % 3 == 2)


def _oct_structure(G, orientation):
    """Group-only label constraints; never construct an alternative lift."""
    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, G.q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("invalid binary-octahedral element order")

    z = (2, 0, 0, 2)
    assert orders[z] == 2
    assert all(g == z for g in G.elements if orders[g] == 2)

    q8 = tuple(g for g in G.elements
               if g in (G.e, z)
               or (_oct_parity(g) == 0 and (g[0]+g[3]) % 3 == 0))
    assert len(q8) == 8

    def powers(g):
        members, value = [], G.e
        while value not in members:
            members.append(value)
            value = G.mul(value, g)
        return tuple(members)

    h4 = powers(next(g for g in q8 if orders[g] == 4))
    h3 = powers(next(g for g in G.elements if orders[g] == 3))
    subgroups = ((G.e, z), h4, h3, q8)
    projections = []
    for subgroup in subgroups:
        unseen, labels = set(G.elements), {}
        while unseen:
            anchor = min(unseen)
            coset = {
                G.mul(anchor, h) if orientation == 0
                else G.mul(h, anchor)
                for h in subgroup
            }
            assert len(coset) == len(subgroup)
            label = min(G.idx(g) for g in coset)
            for g in coset:
                labels[g] = label
            unseen.difference_update(coset)
        projections.append(labels)

    return z, orders, projections


def _extract_components(hx, hz, q):
    """Extract both component maps from the scaffold's actual lifts."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    mask = (1 << q) - 1
    ar = [sum(((row >> (j * cb * q)) & mask) << (j * q)
              for j in range(ca))
          for row in hx[:q]]
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


def _quantum_probe(checks, stabilizers, width, rng, trials):
    """Find genuine logical witnesses, including both sectors.

    Kernel dependencies are classified modulo the exact stabilizer span.
    Dependency sums and addition of check rows preserve commutation.
    A high reading is only an upper bound, never a distance certificate.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    ordered = sorted(basis.items(), reverse=True)

    def logical_class(word):
        for p, row in ordered:
            if (word >> p) & 1:
                word ^= row
        return word

    cols = _component_columns(checks, width)
    order = list(range(width))
    best = width+1
    for _ in range(trials):
        rng.shuffle(order)
        pivots, deps = {}, []
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
                cls = logical_class(word)
                deps.append((word, cls))
                if cls:
                    best = min(best, word.bit_count())

        short = sorted(deps, key=lambda x: x[0].bit_count())[:64]
        seeds = []
        for word, cls in short:
            if cls:
                seeds.append((word, cls))
        for i, (word, cls) in enumerate(short):
            for other, oc in short[:i]:
                if cls ^ oc:
                    v = word ^ other
                    best = min(best, v.bit_count())
                    seeds.append((v, cls ^ oc))
        seeds.sort(key=lambda x: x[0].bit_count())

        # Sparse stabilizer rows explicitly allow cross-sector reduction.
        moves = short + [(row, 0) for row in stabilizers]
        for word, cls in seeds[:4]:
            weight = word.bit_count()
            for sweep in range(3):
                changed = False
                rng.shuffle(moves)
                for other, oc in moves:
                    trial, tc = word ^ other, cls ^ oc
                    tw = trial.bit_count()
                    if tc and tw < weight:
                        word, cls, weight = trial, tc, tw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Binary-octahedral coset cards, retained witnesses, and local repairs."""
    from itertools import combinations

    q = G.q
    assert q == 48
    profile, mode = idx % 5, (idx // 5) % 4
    sizes, counts = _DIC_SUPPORTS[profile]
    sa = [list(sizes)]
    sb = [[col[i] for col in counts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    z, orders, projections = _oct_structure(G, (idx//20) % 2)
    project = projections[mode]
    pools = [[g for g in G.elements if _oct_parity(g) == b]
             for b in (0, 1)]
    mask = (1 << q)-1

    def full_generation(relatives):
        members, seen = [G.e], {G.e}
        for h in members:
            for g in relatives:
                value = G.mul(h, g)
                if value not in seen:
                    seen.add(value)
                    members.append(value)
        return len(members) == q

    def draw(count, side, i, j):
        if not count:
            return []
        if count == 1:
            return [G.pick(rng)]
        if side == "A":
            if mode == 0:
                outer = max(1, count//2-1)
            elif mode == 1:
                outer = 1 if j == 0 else count-1
            elif mode == 2:
                outer = count//2
            else:
                outer = 1 if j == 0 else min(2, count-1)
        elif count == 2:
            outer = (profile+mode+i+j) % 3
        else:
            outer = 1 + ((profile+mode+i+j) % (count-1))

        if mode == 3:
            # Q8 has three cosets in each determinant class.
            # In particular, five-term cells need a 3/2 or 2/3 split.
            outer = max(count-3, min(3, outer))

        quota = [0]*(count-outer) + [1]*outer
        for restart in range(384):
            rng.shuffle(quota)
            cell, used = [], set()
            for b in quota:
                available = []
                for g in pools[b]:
                    if project[g] in used:
                        continue
                    legal = True
                    for h in cell:
                        relative = G.mul(G.inv(h), g)
                        if relative == G.e or relative == z:
                            legal = False
                            break
                        if count == 2:
                            # Even relatives of order six and odd
                            # relatives of order eight are available.
                            wanted = 8 if _oct_parity(relative) else 6
                            if orders[relative] != wanted:
                                legal = False
                                break
                    if legal:
                        available.append(g)
                if not available:
                    break
                g = rng.choice(available)
                cell.append(g)
                used.add(project[g])
            if len(cell) != count:
                continue
            if count >= 3:
                relatives = [G.mul(G.inv(cell[0]), g)
                             for g in cell[1:]]
                if side == "A":
                    if not full_generation(relatives):
                        continue
                elif not any(
                    G.mul(g, h) != G.mul(h, g)
                    for a, g in enumerate(relatives)
                    for h in relatives[:a]
                ):
                    continue
            return cell
        raise ValueError("binary-octahedral coset recipe exhausted")

    def grid(counts, side):
        return [[draw(count, side, i, j) for j, count in enumerate(row)]
                for i, row in enumerate(counts)]

    # Group operations construct test supports only, never another lift.
    subgroup_sets = set()
    for g in G.elements:
        members, v = [], G.e
        while v not in members:
            members.append(v)
            v = G.mul(v, g)
        if len(members) in (2, 3, 4, 6, 8, 12):
            subgroup_sets.add(tuple(sorted(G.idx(h) for h in members)))

    partitions = set()
    for subgroup in subgroup_sets:
        outer = any(_oct_parity(G.elements[h]) for h in subgroup)
        for orientation in range(2):
            unseen, fibers = set(range(q)), []
            while unseen:
                g = G.elements[min(unseen)]
                coset = {
                    G.idx(G.mul(g, G.elements[h]) if orientation == 0
                          else G.mul(G.elements[h], g))
                    for h in subgroup
                }
                unseen.difference_update(coset)
                fibers.append(sum(1 << h for h in coset))
            partitions.add((len(subgroup), tuple(sorted(fibers)), outer))
    partitions = sorted(partitions, reverse=True)

    def restricted(cols, cutoff=12, outer=True, two=False):
        """Find kernels among bounded unions in listed coset partitions.

        All syndromes use verified component columns. This restricted
        search is not an exhaustive quantum-distance test or witness census.
        """
        found = set()
        blocks = len(cols)//q
        for size, fibers, has_outer in partitions:
            if size == 2 and not two:
                continue
            if has_outer and not outer:
                continue
            limit = cutoff//size
            if not limit:
                continue
            words = [fiber << (block*q)
                     for block in range(blocks) for fiber in fibers]
            syndromes = []
            for word in words:
                syndrome = 0
                while word:
                    bit = word & -word
                    syndrome ^= cols[bit.bit_length()-1]
                    word ^= bit
                syndromes.append(syndrome)
            seen = {0: 0}
            for terms in range(1, (limit+1)//2+1):
                for indices in combinations(range(len(words)), terms):
                    syndrome, word = 0, 0
                    for c in indices:
                        syndrome ^= syndromes[c]
                        word ^= words[c]
                    old = seen.get(syndrome)
                    if old is None:
                        seen[syndrome] = word
                    else:
                        witness = word ^ old
                        if witness and witness.bit_count() <= cutoff:
                            found.add(witness)
        return sorted(found, key=lambda w: (w.bit_count(), w))

    def kernel_words(cols, trials, memory=()):
        """Retain actual kernel supports across independent information sets."""
        retained = sorted(set(memory), key=int.bit_count)[:40]
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

            words = sorted(set(retained+dependencies), key=int.bit_count)
            for sweep in range(2):
                reducers = words[:32]
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for other in reducers:
                        combined = word ^ other
                        cw = combined.bit_count()
                        if 0 < cw < weight:
                            word, weight = combined, cw
                    reduced.append(word)
                words = sorted(set(reduced), key=int.bit_count)

            discoveries = set(words+retained)
            short = sorted(discoveries, key=int.bit_count)[:32]
            for a, word in enumerate(short):
                for other in short[:a]:
                    combined = word ^ other
                    if combined:
                        discoveries.add(combined)
            if trial % 4 == 0:
                for a, word in enumerate(short[:8]):
                    for b in range(a):
                        for c in range(b):
                            combined = word ^ short[b] ^ short[c]
                            if combined:
                                discoveries.add(combined)
            retained = sorted(discoveries, key=int.bit_count)[:40]
        return retained

    def reading(words, width):
        return min((w.bit_count() for w in words), default=width+1)

    def akey(item):
        return (min(item["da"], 18), -item["hits"],
                -item["cycles"], q-item["rank"])

    skeleton = [[[G.e] for _ in range(5)] for _ in range(2)]
    a_pool, seen_a = [], set()

    def admit_a(A):
        signature = tuple(tuple(sorted(cell)) for cell in A[0])
        if signature in seen_a:
            return
        seen_a.add(signature)
        hx, hz, _ = lifted_matrices(G, A, skeleton)
        ar, _ = _extract_components(hx, hz, q)
        ac = _component_columns(ar, 2*q)
        structured = restricted(ac)
        aw = kernel_words(ac, 3, structured)
        item = dict(A=A, ac=ac, aw=aw, da=reading(aw, 2*q),
                    rank=gf2_rank(ar),
                    hits=sum(w.bit_count() <= 14 for w in structured),
                    cycles=_component_four_cycles(ac))
        a_pool.append(item)
        a_pool.sort(key=akey, reverse=True)
        del a_pool[8:]

    for attempt in range(96 + 8*(idx//20)):
        admit_a(grid(sa, "A"))

    # Repair an actual witness-heavy A column, not a common relabeling.
    for step in range(40):
        parent = a_pool[rng.randrange(min(4, len(a_pool)))]
        A = [[list(cell) for cell in parent["A"][0]]]
        pressure = [
            1 + sum(((word >> (j*q)) & mask).bit_count()
                    for word in parent["aw"][:8])
            for j in range(2)
        ]
        j = rng.choices(range(2), weights=pressure, k=1)[0]
        A[0][j] = draw(sa[0][j], "A", 0, j)
        if step % 7 == 0:
            A[0][1-j] = draw(sa[0][1-j], "A", 0, 1-j)
        admit_a(A)

    for item in a_pool:
        extra = restricted(item["ac"], cutoff=14, two=True)
        item["aw"] = kernel_words(item["ac"], 32, item["aw"]+extra)
        item["da"] = reading(item["aw"], 2*q)
        item["hits"] = sum(w.bit_count() <= 14 for w in item["aw"])
    a_pool.sort(key=akey, reverse=True)

    finalists, seen_joint = [], set()

    def merit(item):
        # Conservative component heuristics prevent a weak full-code
        # information set from winning merely because it missed a witness.
        d = min(item["da"], item["db"], item.get("logical", 19), 18)
        return (item["k"]*d*d, d, -item["hits"],
                item["da"]+item["db"], -item["cycles"])

    def admit_joint(A, B, ai=None):
        signature = tuple(tuple(sorted(cell))
                          for g in (A, B) for row in g for cell in row)
        if signature in seen_joint:
            return
        seen_joint.add(signature)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (225*n <= 1000*k < 275*n):
            return
        ar, br = _extract_components(hx, hz, q)
        ac = _component_columns(ar, 2*q)
        bc = _component_columns(br, 5*q)
        if ai is None:
            aw = kernel_words(ac, 3, restricted(ac))
        else:
            aw = list(ai["aw"])
        structured = restricted(bc, outer=False)
        bw = kernel_words(bc, 3, structured)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ac, bc=bc, aw=aw, bw=bw,
                    da=reading(aw, 2*q), db=reading(bw, 5*q),
                    hits=sum(w.bit_count() <= 14
                             for w in aw+bw),
                    cycles=_component_four_cycles(ac)
                           + _component_four_cycles(bc))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[8:]

    for attempt in range(112):
        ai = a_pool[attempt % min(4, len(a_pool))]
        if attempt % 14 == 13:
            # Preserve rank diversity if the best classical A strata
            # happen to overshoot the rate gate after coupling to B.
            admit_joint(grid(sa, "A"), grid(sb, "B"))
        else:
            admit_joint(ai["A"], grid(sb, "B"), ai)

    if not finalists:
        raise ValueError("no quotient-separated joint filling in cell 0.250")

    # B repairs target blocks used by retained true kernel witnesses.
    for step in range(32):
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        pressure = [
            1 + sum(((word >> (j*q)) & mask).bit_count()
                    for word in parent["bw"][:8])
            for j in range(5)
        ]
        j = rng.choices(range(5), weights=pressure, k=1)[0]
        occupied = [i for i in range(2) if sb[i][j]]
        i = rng.choice(occupied)
        B[i][j] = draw(sb[i][j], "B", i, j)
        if step % 6 == 0:
            other = (j+1+rng.randrange(4)) % 5
            occupied = [i for i in range(2) if sb[i][other]]
            r = rng.choice(occupied)
            B[r][other] = draw(sb[r][other], "B", r, other)
        ai = dict(aw=parent["aw"])
        admit_joint(parent["A"], B, ai)

    def basis(rows):
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

    def outside(word, pivots):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def dressed(item):
        """Verify embeddings and logical classes, then add stabilizers."""
        best = item["n"]+1
        for side, words, checks, stabilizers in (
                ("A", item["aw"], item["hx"], item["hz"]),
                ("B", item["bw"], item["hz"], item["hx"])):
            pivots = basis(stabilizers)
            seeds = []
            for seed in words[:20]:
                embeddings = (
                    [((seed & mask) << (beta*q))
                     | ((seed >> q) << ((5+beta)*q))
                     for beta in range(5)]
                    if side == "A" else [seed, seed << (5*q)])
                for word in embeddings:
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if outside(word, pivots):
                        best = min(best, word.bit_count())
                        seeds.append(word)

            for word in sorted(set(seeds), key=int.bit_count)[:8]:
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        cw = (word ^ row).bit_count()
                        nearby.append((cw, row))
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    nearby.sort(key=lambda pair: pair[0])
                    moves = [row for _, row in nearby[:16]]
                    for a, row in enumerate(moves):
                        for other in moves[:a]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
        return best

    # Equal, independent rescreening budgets for every retained finalist.
    for item in finalists:
        extra = restricted(item["bc"], two=True)
        item["aw"] = kernel_words(item["ac"], 32, item["aw"])
        item["bw"] = kernel_words(item["bc"], 32, item["bw"]+extra)
        item["da"] = reading(item["aw"], 2*q)
        item["db"] = reading(item["bw"], 5*q)
        item["hits"] = sum(w.bit_count() <= 14
                           for w in item["aw"]+item["bw"])
        item["logical"] = dressed(item)

    finalists.sort(key=merit, reverse=True)
    winner, winner_score = None, None
    for item in finalists[:4]:
        clean = (
            not _small_logical_4(item["hx"], item["hz"], item["n"])
            and not _small_logical_4(item["hz"], item["hx"], item["n"]))
        dx = _quantum_probe(item["hz"], item["hx"], item["n"],
                            rng, trials=12)
        dz = _quantum_probe(item["hx"], item["hz"], item["n"],
                            rng, trials=12)
        item["logical"] = min(item["logical"], dx, dz)
        score = (int(clean),) + merit(item)
        if winner_score is None or score > winner_score:
            winner, winner_score = item, score

    A, B = winner["A"], winner["B"]
    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 225*winner["n"] <= 1000*winner["k"] < 275*winner["n"]
    assert all(len(cell) == len(set(cell))
               for g in (A, B) for row in g for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": winner["k"], "selection": winner_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: C9 monomial voltage rows with isolated odd trinomial upgrades and exact weight<=3 logical rejection did not exceed distance four: all five measured parent codes had k=283 and d30s=4, with minima entirely in sector 1.
# REFUTED: GL(2,3) determinant cards, component probes and exact weight<=4 logical screening did not eliminate weight-12 sector-one logicals: all five measured parent codes had d30s=12.
# REFUTED: Dic12 central-involution exclusion, coset-occupancy cards and joint randomized quantum probes raised the measured minimum above twelve; all five measured current-parent codes instead retained d30s=12 with k=153 and sector-1 minima.
# REFUTED: Three/five-term A rebalancing improves the quotient-separated Dic12 screen over four/four: all twelve current three/five slots read d2s=11, while measured four/four slots reached d30s=13 with sector-one X minima.
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
