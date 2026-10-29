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
CONCEPT = ("C13:C3 with action 3, order 39, shapes (1,2)x(2,5): order-thirteen binomials and C3-norm-free multiplets should produce five measured readings >=14 and exceed parent fitness 43.482 in cell 0.250; measured evaluation can refute this.")
GROUP = ("Jump from Dic12 to the centre-free odd-order Frobenius group C13:C3, y*x*y^-1=x^3. The order-39 nonabelian (2,3)x(2,3) bed reached 45.2, motivating but not validating transfer. This shape bed is untried. Order 39 gives n=468 and removes involutions; all screened matrices still come from verified lifts.")
SHAPES = ("Keep (1,2)x(2,5) for cell 0.250: n=468, kb=117, guaranteed rate 1/4, and component checks/variables 39/78 and 78/195. These shapes allow q<=49. Parent measured minima were weight-thirteen X operators entirely in sector one, across both A columns and multiple B columns; Z readings were fourteen. Exact ranks retain [0.225,0.275).")
FILLING = ("Five revised legal support cards cross four C3 quotient-occupancy policies, including an even-occupancy control. B binomials stay within one quotient coset, so their relative order is thirteen. Larger cells exclude translated C3 norms, require noncommuting relatives and prefer distinct internal ratios. The first three B geometries avoid even/even columns while retaining both row caps at twelve. Verified C3/C13 fiber witnesses, retained kernel words, repairs, exact ranks and stabilizer-aware searches select fresh fillings; restricted exclusions and randomized readings do not certify distance.")
FAMILY_NAME = "r025_F39_12x25_C3normfree_order13_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # C13:C3: x^13=y^3=1, y*x*y^-1=x^3.
    # 3 has multiplicative order three modulo thirteen.
    if not valid_q(q):
        raise ValueError("C13:C3 requires order 39")
    els = [(a,b) for b in range(3) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + pow(3,b,13)*c) % 13, (b+d) % 3)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 3
        return ((-pow(3,bi,13)*a) % 13, bi)

    def idx(g):
        a,b = red(g)
        return 13*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

# Each card is (A cell sizes, B column sizes).
# First three: A row=8, B rows=8/8, B columns<=4:
# HX<=12 and HZ<=12, without any even/even B column.
# Fourth: HX<=11, HZ<=12. Fifth: HX<=10, HZ<=12.
# Every protograph row and column remains nonzero.
_DIC_SUPPORTS = (
    ((4,4), ((3,1), (2,1), (1,2), (1,2), (1,2))),
    ((4,4), ((3,0), (1,3), (2,1), (1,2), (1,2))),
    ((4,4), ((3,1), (1,3), (1,2), (2,1), (1,1))),
    ((3,5), ((2,1), (2,1), (1,2), (1,2), (1,1))),
    ((3,3), ((3,1), (1,3), (2,2), (2,1), (1,2))),
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
    """C3 occupancy cards, order-thirteen pairs, and verified repairs."""
    from itertools import combinations

    q = G.q
    assert q == 39
    profile, mode = idx % 5, (idx // 5) % 4
    sizes, counts = _DIC_SUPPORTS[profile]
    sa = [list(sizes)]
    sb = [[col[i] for col in counts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    pools = [[g for g in G.elements if g[1] == b] for b in range(3)]
    mask = (1 << q)-1

    def draw(count, side, i, j):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            # Distinct elements in one C13 coset have relative order 13.
            b = (profile+mode+i+j) % 3
            return rng.sample(pools[b], 2)

        if side == "A":
            if count == 4:
                recipes = ((3,1,0), (2,2,0), (2,1,1),
                           (3,1,0) if j == 0 else (2,1,1))
            elif count == 3:
                recipes = ((2,1,0), (2,1,0), (1,1,1), (2,0,1))
            else:
                assert count == 5
                recipes = ((3,1,1), (2,2,1), (1,2,2), (2,1,2))
            occupancy = recipes[mode]
        elif count == 3:
            occupancy = ((2,1,0), (1,1,1), (2,0,1),
                         (1,2,0))[(mode+i+j) % 4]
        else:
            assert count == 4
            occupancy = ((3,1,0), (2,1,1), (3,0,1),
                         (1,2,1))[(mode+i+j) % 4]

        best = None
        for restart in range(96):
            phase = rng.randrange(3)
            direction = rng.choice((1,2))
            cell = []
            for b, amount in enumerate(occupancy):
                cell.extend(rng.sample(
                    pools[(phase+direction*b) % 3], amount))
            assert len(cell) == len(set(cell)) == count

            # Exclude a translated norm of any order-three subgroup.
            # This is a label filter, not a separately constructed lift.
            bad = False
            for triple in combinations(cell, 3):
                anchor = G.inv(triple[0])
                u = G.mul(anchor, triple[1])
                v = G.mul(anchor, triple[2])
                if u[1] and v == G.mul(u, u):
                    bad = True
                    break
            if bad:
                continue

            anchor = G.inv(cell[0])
            relatives = [G.mul(anchor, g) for g in cell[1:]]
            if not any(G.mul(g,h) != G.mul(h,g)
                       for a, g in enumerate(relatives)
                       for h in relatives[:a]):
                continue

            ratios = [G.mul(g, G.inv(h))
                      for g in cell for h in cell if g != h]
            cost = len(ratios)-len(set(ratios))
            if best is None or cost < best[0]:
                best = (cost, cell)
            if cost == 0:
                break
        if best is None:
            raise ValueError("C3-norm-free occupancy recipe exhausted")
        return best[1]

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
        if len(members) in (3, 13):
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

    def restricted(cols, cutoff=13, outer=True, two=False):
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
                    hits=sum(w.bit_count() <= 13 for w in structured),
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
                    hits=sum(w.bit_count() <= 13
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
        item["hits"] = sum(w.bit_count() <= 13
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
