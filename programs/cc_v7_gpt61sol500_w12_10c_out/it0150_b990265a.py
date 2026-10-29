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
CONCEPT = ("GL(2,3), order 48, A=(1,2), B=(2,5): reflection-pair-controlled quartets, central-pair dependency screening and positive rank-surplus selection should exceed fitness 38.25 in cell 0.250; measured evaluation can refute this.")
GROUP = ("Frozen under L1. This exact bed produced five measured k=153, d30s=12 codes. A related higher-rank filling produced several k=156, d30s=12 codes but also a mixed-sector weight-eleven witness. GL(2,3)'s central involution permits a restricted, exact paired-support screen without constructing a different lift.")
SHAPES = ("Frozen (1,2)x(2,5): n=576, kb=144, guaranteed rate 1/4, maximum admissible lift order 49 and actual order 48. Component variable/check counts are 96/48 and 240/96. Every measured parent minimum was sector one, across both A columns and multiple B columns; no parent minimum used Arow/Brow blocks. Exact full ranks retain [0.225,0.275).")
FILLING = ("Five legal B incidence profiles cross four exact A reflection-pair budgets: (0,0), (0,1), (1,1), (1,2). Exclude central-involution-separated terms but allow noncentral relative involutions. Screen verified A components for central-paired dependencies through weight ten and prefer useful rank-44 components rather than penalizing surplus. Optimize fresh B fillings, then test component embeddings and full mixed-sector witnesses modulo exact stabilizer spans. Restricted exclusions are not quantum-distance certificates; randomized readings remain upper bounds.")
FAMILY_NAME = "r025_GL2F3_12x25_reflection_centralpair_rankbonus_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # GL(2,F3), encoded by row-major entries (a,b,c,d).
    if not valid_q(q):
        raise ValueError("GL(2,F3) has order 48")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = red(g)
        determinant = (a*d-b*c) % 3
        # Each nonzero element of F3 is its own multiplicative inverse.
        return ((determinant*d) % 3, (-determinant*b) % 3,
                (-determinant*c) % 3, (determinant*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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
    """Restrict the VERIFIED lift; do not implement regular actions."""
    mask = (1 << q) - 1
    cb = B_SHAPE[1]
    ar = [
        (row & mask) | (((row >> (cb*q)) & mask) << q)
        for row in hx[:q]
    ]
    bm = (1 << (cb*q)) - 1
    br = [row & bm for row in hz[:B_SHAPE[0]*q]]
    return ar, br, _columns(ar, 2*q), _columns(br, cb*q)


def NEW_CTX(rng, G, idx):
    """Rank-bonus search with new reflection budgets and verified actions."""
    q = G.q
    assert q == 48
    recipe, mode = idx % 20, idx % 4
    incidence = _SUPPORT_PLANS[recipe // 4]
    sa = [[4, 4]]
    sb = [[column[i] for column in incidence] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    els = G.elements
    orders = _element_orders(G)
    z = (2, 0, 0, 2)
    relative = {
        (g, h): G.mul(G.inv(g), h)
        for g in els for h in els
    }
    reflection = {
        (g, h): orders[relative[g, h]] == 2 and relative[g, h] != z
        for g in els for h in els
    }
    budgets = ((0, 0), (0, 1), (1, 1), (1, 2))[mode]

    def draw(count, wanted=None, binomial_order=None):
        if count <= 1:
            return [] if not count else [G.pick(rng)]
        for restart in range(768):
            cell = rng.sample(els, count)
            pairs = [(cell[a], cell[b])
                     for a in range(count) for b in range(a)]
            if any(relative[g, h] == z for g, h in pairs):
                continue
            if wanted is not None:
                if sum(reflection[g, h] for g, h in pairs) != wanted:
                    continue
            if binomial_order is not None:
                if orders[relative[cell[0], cell[1]]] != binomial_order:
                    continue
            if count >= 3:
                rel = [relative[cell[0], g] for g in cell[1:]]
                if not any(G.mul(g, h) != G.mul(h, g)
                           for a, g in enumerate(rel) for h in rel[:a]):
                    continue
            return cell
        raise ValueError("reflection-pair recipe exhausted")

    def draw_b(attempt):
        grid = []
        for i, row in enumerate(sb):
            out = []
            for j, count in enumerate(row):
                if count == 2:
                    wanted_order = (6, 8)[(mode+i+j+idx//20) % 2]
                    cell = draw(count, binomial_order=wanted_order)
                elif count >= 3:
                    # These multiplets can contain noncentral relative
                    # involutions, unlike the parent's universal exclusion.
                    wanted = (mode+i+j+attempt//16) % 2
                    cell = draw(count, wanted=wanted)
                else:
                    cell = draw(count)
                out.append(cell)
            grid.append(out)
        return grid

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
        return tuple(sorted(pivots.items(), reverse=True))

    def residue(word, span):
        for p, row in span:
            if (word >> p) & 1:
                word ^= row
        return word

    def probe(cols, trials, classify=None, initial=None, moves=(),
              full=False):
        """Find actual kernel or logical witnesses, never certificates.

        classify tests exact stabilizer membership of verified embeddings.
        An initial ceiling is used only when already supplied by an actual
        logical witness from the same code.
        """
        best = len(cols)+1 if initial is None else initial
        found, memory = set(), []
        order = list(range(len(cols)))

        def observe(word):
            nonlocal best, found
            if not word:
                return
            weight = word.bit_count()
            if weight > best:
                return
            if classify is not None and not classify(word):
                return
            if weight < best:
                best, found = weight, {word}
            else:
                found.add(word)

        for trial in range(trials):
            rng.shuffle(order)
            if full and trial % 3 != 1:
                # Different information sets deliberately expose sector-2
                # preimages, rather than only sector-one dependencies.
                first_sector_two = trial % 3 == 0
                order.sort(key=lambda c:
                           (c < 10*q) if first_sector_two else (c >= 10*q))
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
                short = words[:32]
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for other in short:
                        candidate = word ^ other
                        cw = candidate.bit_count()
                        if 0 < cw < weight:
                            word, weight = candidate, cw
                    reduced.append(word)
                    observe(word)
                words = sorted(set(reduced), key=int.bit_count)

            discoveries = set(words)
            for a, word in enumerate(words[:32]):
                for other in words[:a]:
                    candidate = word ^ other
                    if candidate:
                        observe(candidate)
                        discoveries.add(candidate)

            if classify is not None:
                seeds = [word for word in raw if classify(word)][:6]
                local_moves = words + list(moves)
                pair_moves = [a ^ b for t, a in enumerate(words[:12])
                              for b in words[:t] if a != b]
                for word in seeds:
                    weight = word.bit_count()
                    for sweep in range(3):
                        changed = False
                        rng.shuffle(local_moves)
                        for other in local_moves:
                            candidate = word ^ other
                            cw = candidate.bit_count()
                            if cw < weight and candidate and classify(candidate):
                                word, weight, changed = candidate, cw, True
                        if not changed:
                            for other in pair_moves:
                                candidate = word ^ other
                                cw = candidate.bit_count()
                                if (0 < cw < weight and classify(candidate)):
                                    word, weight, changed = candidate, cw, True
                                    break
                        observe(word)
                        if not changed:
                            break
                    discoveries.add(word)

            memory = sorted(discoveries, key=int.bit_count)[:48]
            if best <= 4:
                break
        return best, len(found)

    # Central pairs are label constraints only. Their column syndromes
    # are taken from the verified A lift, not a handmade regular action.
    central_pairs = []
    unseen = set(range(q))
    while unseen:
        a = min(unseen)
        b = G.idx(G.mul(z, els[a]))
        assert a != b
        unseen.difference_update((a, b))
        central_pairs.append((a, b))

    def paired_profile(ac):
        """Exact restricted exclusion through actual weight ten.

        Enumerating subsets of <=3 paired columns also finds weight-twelve
        witnesses. Sentinel 14 means only that this restricted screen found
        none through twelve. It is never promoted as a distance reading.
        """
        cols = [ac[block*q+a] ^ ac[block*q+b]
                for block in range(2) for a, b in central_pairs]
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is not None:
                return 2*(word ^ old).bit_count(), 0
            seen[syndrome] = word
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is not None:
                    return 2*(word ^ old).bit_count(), 0
                seen[syndrome] = word

        hits = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    word = (1 << i) | (1 << j) | (1 << h)
                    old = seen.get(syndrome)
                    if old is not None:
                        weight = (word ^ old).bit_count()
                        if weight <= 5:
                            return 2*weight, 0
                        hits += 1
                    else:
                        seen[syndrome] = word
        # This is a collision tie-break, not an exhaustive multiplicity.
        return (12, hits) if hits else (14, 0)

    skeleton = [[[G.e] for _ in range(5)] for _ in range(2)]
    strata = {}
    for attempt in range(256 + 24*(idx//20)):
        try:
            A = [[draw(4, wanted=budgets[j]) for j in range(2)]]
        except ValueError:
            continue
        hx, hz, _ = lifted_matrices(G, A, skeleton)
        ar, _, ac, _ = _component_data(hx, hz, q)
        rank = gf2_rank(ar)
        if not 44 <= rank <= 47:
            continue
        floor = _short_floor(ac)
        paired, hits = paired_profile(ac)
        cycles = _component_cycles(ar)
        item = dict(A=A, ac=ac, rank=rank, floor=floor,
                    paired=paired, hits=hits, cycles=cycles)
        bucket = strata.setdefault(rank, [])
        bucket.append(item)
        bucket.sort(key=lambda c: (
            c["floor"], c["paired"] >= 12, -c["hits"], -c["cycles"]),
            reverse=True)
        del bucket[4:]

    a_pool = [item for bucket in strata.values() for item in bucket]
    if not a_pool:
        raise ValueError("no ranked reflection-budget A component")

    for item in a_pool:
        da, _ = probe(item["ac"], 8)
        if item["paired"] <= 12:
            da = min(da, item["paired"])
        if item["floor"] < 5:
            da = min(da, item["floor"])
        item["da"] = da

    def a_merit(item):
        # This rank-surplus expression is a prioritization heuristic,
        # NOT an asserted formula for the full nonabelian LP dimension.
        rank_priority = 144 + 3*(q-item["rank"])
        d = min(item["da"], 16)
        return (item["floor"], rank_priority*d*d, d,
                -item["hits"], -item["cycles"])

    a_pool.sort(key=a_merit, reverse=True)
    a_pool = a_pool[:4]
    for item in a_pool:
        da, _ = probe(item["ac"], 40)
        item["da"] = min(item["da"], da)
    a_pool.sort(key=a_merit, reverse=True)

    def merit(item):
        d = min(item["da"], item["db"], 16)
        return (item["floor"], item["k"]*d*d, d,
                item["da"]+item["db"], -item["cycles"])

    finalists = []
    for attempt in range(80 + 8*(idx//20)):
        ai = a_pool[attempt % len(a_pool)]
        try:
            B = draw_b(attempt)
        except ValueError:
            continue
        A = ai["A"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k > 0 and 225*n <= 1000*k < 275*n):
            continue
        ar, br, ac, bc = _component_data(hx, hz, q)
        fb = _short_floor(bc)
        db, _ = probe(bc, 4)
        if fb < 5:
            db = min(db, fb)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ac, bc=bc, da=ai["da"], db=db,
                    floor=min(ai["floor"], fb),
                    cycles=ai["cycles"]+_component_cycles(br))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[5:]

    if not finalists:
        raise ValueError("no joint reflection-budget filling in cell 0.250")

    mask = (1 << q)-1

    def embed_a(word):
        return tuple(((word & mask) << (beta*q)) |
                     ((word >> q) << ((5+beta)*q))
                     for beta in range(5))

    def embed_b(word):
        return (word, word << (5*q))

    def low4(cols, span):
        """Exact full-code logical existence test through weight four."""
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif residue(word ^ old, span):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif residue(word ^ old, span):
                    return True
        return False

    for item in finalists:
        xs, zs = basis(item["hx"]), basis(item["hz"])
        item["xs"], item["zs"] = xs, zs
        item["xc"] = _columns(item["hz"], item["n"])
        item["zc"] = _columns(item["hx"], item["n"])
        ca = lambda w, span=zs: any(residue(v, span) for v in embed_a(w))
        cb = lambda w, span=xs: any(residue(v, span) for v in embed_b(w))
        da, ha = probe(item["ac"], 36, ca)
        db, hb = probe(item["bc"], 28, cb)
        item["da"], item["db"], item["hits"] = da, db, ha+hb
        item["clean"] = (not low4(item["xc"], xs)
                         and not low4(item["zc"], zs))
        item["witnessed"] = da <= 2*q and db <= 5*q

    viable = [item for item in finalists if item["witnessed"]]
    if not viable:
        # Keep a fresh, legal fallback; never score a no-witness sentinel.
        winner = max(finalists, key=lambda c: (
            c["clean"], c["floor"], c["k"], -c["cycles"]))
    else:
        viable.sort(key=lambda c: (c["clean"],)+merit(c), reverse=True)
        for item in viable[:3]:
            cx = lambda w, span=item["xs"]: bool(residue(w, span))
            cz = lambda w, span=item["zs"]: bool(residue(w, span))
            dx, hx_hits = probe(
                item["xc"], 18, cx, item["db"], item["hx"], full=True)
            dz, hz_hits = probe(
                item["zc"], 18, cz, item["da"], item["hz"], full=True)
            item["dx"], item["dz"] = dx, dz
            item["hits"] += hx_hits+hz_hits

        def final_merit(item):
            d = min(item["dx"], item["dz"], 16)
            return (item["clean"], item["k"]*d*d, d,
                    -item["hits"], -item["cycles"])
        winner = max(viable[:3], key=final_merit)

    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 225*winner["n"] <= 1000*winner["k"] < 275*winner["n"]
    assert all(len(cell) == len(set(cell))
               for grid in (winner["A"], winner["B"])
               for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": winner["A"], "B": winner["B"]},
            "k": winner["k"]}


def CELL(rng, G, side, i, j, ctx):
    """All labels and counts were selected jointly in NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting involution-free GL(2,3) quartets with constituent screening through six eliminate sector-one weight-twelve logicals; all five measured parent codes instead had k=153, d30s=12, with both minimum types entirely in sector one.
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
