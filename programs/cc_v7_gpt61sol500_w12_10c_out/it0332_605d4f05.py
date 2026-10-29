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
CONCEPT = ("C23 x C2, cyclic of order 46, shapes (2,3)x(2,3): relative-order-controlled binomials and asymmetric trinomial upgrades selected against retained verified witnesses should exceed five-code fitness 60.3 in cell 0.100; measured evaluation can refute this.")
GROUP = ("Jump from Dic12 to the abelian C23 x C2, isomorphic to C46. This genuinely different group has a measured (2,3)x(2,3) bed record of 60.3 with distances 24-28, compared with the parent's 43.482 and distance thirteen. Reuse the promising group/shape bed, not an archived filling or an assumed distance advantage.")
SHAPES = ("Move to equal (2,3) sides for cell 0.100: n=13*46=598, kb=46, guaranteed rate 1/13, and equal constituent checks/variables 92/138. The size window permits q<=46 and this group saturates it. Parent weight-thirteen minima occupied sector-one Acol/Bcol blocks broadly; independently optimize both new constituents. Exact full ranks require positive k and k/n<0.125.")
FILLING = ("Five count architectures cross four intrinsic relative-order/AP policies. Counts are one to three; every row totals at most seven and every column at most five, so both CSS row caps are twelve. Compile constituent actions exclusively from verified unit-cell lifts, screen short dependencies, retain actual kernel words across information sets, and repair witness-heavy columns. Verify quantum embeddings modulo stabilizers, shorten by stabilizer additions, and independently probe both full matrices. Unequal uncoloured Tanner fingerprints separate emitted designs; all randomized readings remain upper estimates, not distance certificates.")
FAMILY_NAME = "r010_C23xC2_23x23_order_AP_retained_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Direct product C23 x C2, hence cyclic of order 46.
    if not valid_q(q):
        raise ValueError("C23 x C2 requires order 46")
    els = [(a,b) for b in range(2) for a in range(23)]

    def red(g):
        return (g[0] % 23, g[1] % 2)

    def mul(g, h):
        return ((g[0]+h[0]) % 23, (g[1]+h[1]) % 2)

    def inv(g):
        return ((-g[0]) % 23, (-g[1]) % 2)

    def idx(g):
        a,b = red(g)
        return 23*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)

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


def _dic_cell(rng, G, pools, count, outer, long_binomial):
    """Prescribed coset occupancy; exclude the unique involution difference."""
    from math import gcd
    if count == 0:
        return []
    quota = [0]*(count-outer) + [1]*outer
    for _ in range(64):
        rng.shuffle(quota)
        cell = []
        for coset in quota:
            available = []
            for g in pools[coset]:
                legal = True
                for h in cell:
                    relative = G.mul(G.inv(h), g)
                    if relative == G.e or relative == (12,0):
                        legal = False
                        break
                    if long_binomial and count == 2 and relative[1] == 0:
                        if 24 // gcd(relative[0], 24) < 8:
                            legal = False
                            break
                if legal:
                    available.append(g)
            if not available:
                break
            cell.append(rng.choice(available))
        if len(cell) == count:
            return cell
    raise ValueError("dicyclic cell rejection exhausted")


def _dic_grid(rng, G, pools, counts, mode, side, phase):
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, count in enumerate(row):
            if count <= 1:
                outer = count*((phase+i+j) % 2)
            elif side == "A":
                if mode == 0:
                    outer = count//2 + (count % 2)*(j % 2)
                elif mode == 1:
                    outer = 1
                elif mode == 2:
                    outer = 1 if j == 0 else count-1
                else:
                    outer = count//2 if j == 0 else count-1
            elif mode == 0:
                outer = (count*((i+j+phase) % 2) if count == 2
                         else count//2 + (count % 2)*((i+j+phase) % 2))
            elif mode == 1:
                outer = 1
            elif mode == 2:
                outer = 0 if count == 2 else count-1
            else:
                outer = (count*((i+j) % 2) if count == 2
                         else rng.randrange(count+1))
            out.append(_dic_cell(rng, G, pools, count, outer,
                                 side == "B" and mode in (0,2,3)))
        grid.append(out)
    return grid


def _extract_components(hx, hz, q):
    """Extract every constituent row from the scaffold's actual lifts."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = [
        sum(((hx[(i*cb)*q+t] >> (j*cb*q)) & mask) << (j*q)
            for j in range(ca))
        for i in range(ra) for t in range(q)
    ]
    br = [row & ((1 << (cb*q))-1) for row in hz[:rb*q]]
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


def _retired_dic12_context(rng, G, idx):
    """Inactive historical Dic12 optimizer; the new NEW_CTX never calls it."""
    from itertools import combinations
    from math import gcd

    q = G.q
    assert q == 48
    profile, mode = idx % 5, (idx // 5) % 4
    sizes, counts = _DIC_SUPPORTS[profile]
    sa = [list(sizes)]
    sb = [[col[i] for col in counts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    modulus = (8, 4, 6, 3)[mode]
    pools = [[g for g in G.elements if g[1] == b] for b in (0, 1)]
    mask = (1 << q)-1

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

        # Projected supports differ between cards, not just their translates.
        quota = [0]*(count-outer) + [1]*outer
        for restart in range(96):
            rng.shuffle(quota)
            cell, used = [], set()
            for b in quota:
                available = []
                for g in pools[b]:
                    if (g[0] % modulus, b) in used:
                        continue
                    legal = True
                    for h in cell:
                        relative = G.mul(G.inv(h), g)
                        if relative == G.e or relative == (12, 0):
                            legal = False
                            break
                        if count == 2 and relative[1] == 0:
                            if 24 // gcd(relative[0], 24) < 6:
                                legal = False
                                break
                    if legal:
                        available.append(g)
                if not available:
                    break
                g = rng.choice(available)
                cell.append(g)
                used.add((g[0] % modulus, b))
            if len(cell) == count:
                return cell
        raise ValueError("quotient-separated cell recipe exhausted")

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
        outer = any(G.elements[h][1] for h in subgroup)
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
                    hits=sum(w.bit_count() == 12 for w in structured),
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
        extra = restricted(item["ac"], two=True)
        item["aw"] = kernel_words(item["ac"], 32, item["aw"]+extra)
        item["da"] = reading(item["aw"], 2*q)
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
                    hits=sum(w.bit_count() == 12
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
        item["hits"] = sum(w.bit_count() == 12
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


# New equal-side count architectures. Every row is <=7 and column <=5.
# Cards differ in odd-cell number/location and in degree-one asymmetry.
_C46_COUNTS = (
    (((2,2,2), (2,2,3)), ((2,2,3), (2,3,2))),
    (((2,2,2), (2,2,3)), ((2,2,2), (2,3,2))),
    (((2,2,3), (2,3,2)), ((2,2,3), (2,3,2))),
    (((2,2,3), (2,3,2)), ((3,2,2), (2,2,3))),
    (((3,1,3), (1,3,2)), ((2,2,3), (2,3,2))),
)


@__import__("functools").lru_cache(maxsize=1)
def _c46_atlas(G):
    """Compile unit actions only through the verified construction helper."""
    q = G.q
    mask = (1 << q)-1
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[] for _ in range(3)] for _ in range(2)]
            B = [[[] for _ in range(3)] for _ in range(2)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            block = _component_columns(
                [row & mask for row in source[:q]], q)
            assert all(s.bit_count() == 1 for s in block)
            blocks.append(tuple(block))
        atlas[side] = tuple(blocks)
    return atlas


def _c46_columns(G, atlas, side, grid):
    """F2-linear superposition of scaffold-generated unit blocks."""
    q = G.q
    columns = [0]*(3*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[side][G.idx(g)]
                for t, syndrome in enumerate(block):
                    columns[j*q+t] ^= syndrome << (i*q)
    return columns


def _c46_draw(rng, G, counts, mode, side, lane):
    """Intrinsic order-23/order-46 and AP/non-AP policies, not orbit copies."""
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, count in enumerate(row):
            if count == 1:
                out.append([G.pick(rng)])
                continue
            cross = (
                1 if mode == 0 else
                0 if mode == 1 else
                (i+j+int(side == "B")+lane) % 2
            )
            if count == 2:
                base = G.pick(rng)
                # Nonzero C23 displacement gives relative order
                # 23 for cross=0 and 46 for cross=1.
                delta = (rng.randrange(1,23), cross)
                out.append([base, G.mul(base, delta)])
                continue
            if mode == 3:
                base = G.pick(rng)
                delta = (rng.randrange(1,23), cross)
                out.append([
                    base, G.mul(base, delta),
                    G.mul(G.mul(base, delta), delta)])
                continue

            # Non-AP projected triples, with a mixed C2 occupancy.
            # Distinct C23 coordinates also exclude involution differences.
            for attempt in range(128):
                aa = rng.sample(range(23), 3)
                if any(
                    (aa[b]+aa[c]-2*aa[a]) % 23 == 0
                    for a,b,c in ((0,1,2), (1,0,2), (2,0,1))
                ):
                    continue
                outer = (
                    2 if mode == 0 else
                    1 if mode == 1 else
                    1+(i+j+int(side == "B")+lane) % 2)
                bb = [0]*(3-outer)+[1]*outer
                rng.shuffle(bb)
                out.append(list(zip(aa, bb)))
                break
            else:
                raise ValueError("non-AP triple recipe exhausted")
        grid.append(out)
    assert all(len(c) == len(set(c)) <= 3
               for row in grid for c in row)
    return grid


def _c46_words(cols, rng, trials, seeds=()):
    """Retain actual nonzero kernel supports across information sets."""
    memory = sorted({w for w in seeds if w},
                    key=lambda w: (w.bit_count(), w))[:48]
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

        discoveries = set(memory)
        words = sorted(set(dependencies+memory),
                       key=lambda w: (w.bit_count(), w))
        discoveries.update(words[:48])
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
            discoveries.update(words[:48])

        small = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:32]
        for i, word in enumerate(small):
            for other in small[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        if trial % 4 == 0:
            for i, word in enumerate(small[:10]):
                for j in range(i):
                    for h in range(j):
                        candidate = word ^ small[j] ^ small[h]
                        if candidate:
                            discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:48]
        if memory and memory[0].bit_count() <= 4:
            break
    assert memory
    return tuple(memory)


def _c46_pool(rng, G, atlas, side, counts, mode, lane, quota):
    """Optimize independently while keeping several constituent rank strata."""
    raw, seen = {}, set()
    q = G.q

    def admit(grid):
        labels = tuple(tuple(sorted(c)) for row in grid for c in row)
        if labels in seen:
            return
        seen.add(labels)
        cols = _c46_columns(G, atlas, side, grid)
        rank = gf2_rank(cols)
        if rank < 2*q-6:
            return
        cycles = _component_four_cycles(cols)
        item = dict(grid=grid, cols=cols, rank=rank, cycles=cycles)
        bucket = raw.setdefault(rank, [])
        bucket.append(item)
        bucket.sort(key=lambda x: x["cycles"])
        del bucket[3:]

    for attempt in range(quota):
        admit(_c46_draw(rng, G, counts, mode, side, lane))

    pool = []
    for bucket in raw.values():
        for item in bucket:
            floor, _ = _short_kernel(item["cols"])
            if floor < 5:
                continue
            item["words"] = _c46_words(item["cols"], rng, 4)
            item["d"] = item["words"][0].bit_count()
            pool.append(item)
    if not pool:
        return []

    def key(item):
        return (min(item["d"],36), -item["cycles"], item["rank"])

    pool.sort(key=key, reverse=True)
    pool = pool[:10]
    mask = (1 << q)-1

    # Replace labels in columns actually used by retained short words.
    for step in range(24 if side == "A" else 40):
        parent = rng.choice(pool[:min(4,len(pool))])
        grid = [[list(c) for c in row] for row in parent["grid"]]
        pressure = [
            1+sum(((word >> (j*q)) & mask).bit_count()
                  for word in parent["words"][:12])
            for j in range(3)]
        j = rng.choices(range(3), weights=pressure, k=1)[0]
        replacement = _c46_draw(rng, G, counts, mode, side, lane)
        i = rng.randrange(2)
        grid[i][j] = replacement[i][j]
        if step % 5 == 0:
            grid[1-i][j] = replacement[1-i][j]
        labels = tuple(tuple(sorted(c)) for row in grid for c in row)
        if labels in seen:
            continue
        seen.add(labels)
        cols = _c46_columns(G, atlas, side, grid)
        rank = gf2_rank(cols)
        if rank < 2*q-6 or _short_kernel(cols)[0] < 5:
            continue
        words = _c46_words(cols, rng, 4)
        pool.append(dict(
            grid=grid, cols=cols, rank=rank, words=words,
            d=words[0].bit_count(), cycles=_component_four_cycles(cols)))
        # Keep a representative of each observed rank before spare places.
        ordered = sorted(pool, key=key, reverse=True)
        selected, ranks = [], set()
        for item in ordered:
            if item["rank"] not in ranks:
                selected.append(item)
                ranks.add(item["rank"])
        selected = selected[:10]
        for item in ordered:
            if len(selected) >= 10:
                break
            if not any(item is old for old in selected):
                selected.append(item)
        pool = selected

    for item in pool:
        item["words"] = _c46_words(
            item["cols"], rng, 28, item["words"])
        item["d"] = min(item["d"], item["words"][0].bit_count())
    pool.sort(key=key, reverse=True)
    # Six constituent alternatives leave room for exact coupled rate gates.
    return pool[:6]


def _c46_span(rows):
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


def _c46_outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c46_embedded_reading(item, q):
    """Verify constituent embeddings as genuine logicals, then dress them."""
    mask = (1 << q)-1
    best = item["n"]+1
    for side, words, checks, stabilizers in (
            ("A", item["aa"]["words"], item["hx"], item["hz"]),
            ("B", item["bb"]["words"], item["hz"], item["hx"])):
        span = _c46_span(stabilizers)
        seeds = []
        for compact in words[:24]:
            embeddings = (
                [sum(((compact >> (j*q)) & mask)
                     << ((j*3+beta)*q) for j in range(3))
                 for beta in range(3)]
                if side == "A" else
                [compact << (j*3*q) for j in range(3)])
            for word in embeddings:
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if _c46_outside(word, span):
                    best = min(best, word.bit_count())
                    seeds.append(word)

        for word in sorted(set(seeds), key=int.bit_count)[:8]:
            for sweep in range(3):
                weight, chosen = word.bit_count(), word
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda x: x[0])
                for cw, row in nearby:
                    if cw < weight:
                        weight, chosen = cw, word ^ row
                moves = [row for _,row in nearby[:16]]
                for a, row in enumerate(moves):
                    for other in moves[:a]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            weight, chosen = cw, candidate
                if chosen == word:
                    break
                word = chosen
                best = min(best, weight)
    return best


def _c46_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariants; equality is conservatively rejected."""
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [r.bit_count() for r in rows]
    adjacency = [0]*len(rows)
    local = [Counter() for _ in rows]
    overlaps, edges = Counter(), []
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] += 1
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                local[i][(weights[j],overlap)] += 1
                local[j][(weights[i],overlap)] += 1
                edges.append((i,j,overlap))
    common = Counter(
        (min(weights[i],weights[j]), max(weights[i],weights[j]),
         overlap, (adjacency[i] & adjacency[j]).bit_count())
        for i,j,overlap in edges)
    profiles = Counter(
        (weights[i],tuple(sorted(profile.items())))
        for i,profile in enumerate(local))
    degrees = Counter(
        c.bit_count() for c in _component_columns(rows,n))
    return (
        n, tuple(sorted(Counter(weights).items())),
        tuple(sorted(degrees.items())), tuple(sorted(overlaps.items())),
        tuple(sorted(profiles.items())), tuple(sorted(common.items())))


@__import__("functools").lru_cache(maxsize=1)
def _c46_bank(G):
    """Twenty deliberate structural tasks; no archived or transported seed."""
    assert G.q == 46 and A_SHAPE == B_SHAPE == (2,3)
    q = G.q
    atlas = _c46_atlas(G)
    bank, emitted = [], set()

    def merit(item):
        # Constituent words are conservative heuristics until an embedding
        # has passed commutation and exact stabilizer-membership tests.
        d = min(item["aa"]["d"], item["bb"]["d"],
                item["logical"], 36)
        return (item["k"]*d*d/item["n"], d,
                min(item["logical"],36), -item["cycles"])

    for slot in range(N_SLOTS):
        profile, mode = slot % 5, (slot//5) % 4
        lane = slot//20
        sa, sb = _C46_COUNTS[profile]
        assert _grid_weights(sa,sb) <= W_MAX
        assert _no_dead_lines(sa,sb)
        rng = random.Random(
            BASE_SEED*100003 + 461002303 + slot*104729)
        winner = None

        for retry in range(2):
            ap = _c46_pool(
                rng,G,atlas,"A",sa,mode,lane,112+64*retry)
            bp = _c46_pool(
                rng,G,atlas,"B",sb,mode,lane,144+64*retry)
            proposals = []
            for aa in ap:
                for bb in bp:
                    hx,hz,n = lifted_matrices(G,aa["grid"],bb["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (k > 0 and 1000*k < 125*n):
                        continue
                    ar,br = _extract_components(hx,hz,q)
                    assert _component_columns(ar,3*q) == aa["cols"]
                    assert _component_columns(br,3*q) == bb["cols"]
                    d = min(aa["d"],bb["d"],36)
                    proposals.append(dict(
                        A=aa["grid"],B=bb["grid"],aa=aa,bb=bb,
                        hx=hx,hz=hz,n=n,k=k,
                        cycles=aa["cycles"]+bb["cycles"],
                        key=(k*d*d/n,d,-aa["cycles"]-bb["cycles"])))
            proposals.sort(key=lambda x:x["key"],reverse=True)

            finalists, local = [], set()
            for item in proposals:
                hx,hz,n = item["hx"],item["hz"],item["n"]
                if (_small_logical_4(hx,hz,n)
                        or _small_logical_4(hz,hx,n)):
                    continue
                fp = _c46_fingerprint(hx,hz,n)
                if fp in emitted or fp in local:
                    continue
                local.add(fp)
                item["fp"] = fp
                item["logical"] = _c46_embedded_reading(item,q)
                finalists.append(item)
                if len(finalists) >= 4:
                    break

            # Equal independent full-code budgets; earlier readings survive.
            for item in finalists:
                dx = _quantum_probe(
                    item["hz"],item["hx"],item["n"],rng,trials=16)
                dz = _quantum_probe(
                    item["hx"],item["hz"],item["n"],rng,trials=16)
                item["logical"] = min(item["logical"],dx,dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 26:
                break

        if winner is None:
            bank.append(None)
            continue
        hx,hz = winner["hx"],winner["hz"]
        assert winner["n"] == 598
        assert 46 <= winner["k"] and 1000*winner["k"] < 125*598
        assert css_ok(hx,hz)
        assert max(r.bit_count() for r in hx+hz) <= W_MAX
        assert all(
            len(c) == len(set(c)) <= 3
            for grid in (winner["A"],winner["B"])
            for row in grid for c in row)
        emitted.add(winner["fp"])
        freeze = lambda grid: tuple(
            tuple(tuple(c) for c in row) for row in grid)
        bank.append((
            freeze(winner["A"]),freeze(winner["B"]),
            winner["k"],merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this independently optimized, rank-verified structural card."""
    selected = _c46_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh screened C46 filling in cell 0.100")
    A,B,k,score = selected
    return {"idx":idx,"cells":{"A":A,"B":B},
            "k":k,"selection":score}


def CELL(rng, G, side, i, j, ctx):
    """Return every stored term of the new equal-side construction."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: C9 monomial voltage rows with isolated odd trinomial upgrades and exact weight<=3 logical rejection did not exceed distance four: all five measured parent codes had k=283 and d30s=4, with minima entirely in sector 1.
# REFUTED: GL(2,3) determinant cards, component probes and exact weight<=4 logical screening did not eliminate weight-12 sector-one logicals: all five measured parent codes had d30s=12.
# REFUTED: Dic12 central-involution exclusion, coset-occupancy cards and joint randomized quantum probes raised the measured minimum above twelve; all five measured current-parent codes instead retained d30s=12 with k=153 and sector-1 minima.
# REFUTED: Three/five A rebalancing improves the quotient-separated Dic12 screen over four/four: every distinct current-parent three/five slot read d2s=11, while the measured four/four leaders reached d30s=13 with sector-one minima.
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
