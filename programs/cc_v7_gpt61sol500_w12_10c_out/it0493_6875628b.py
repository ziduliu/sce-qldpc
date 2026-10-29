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
CONCEPT = ("C16 semidirect C2 with action 7, the semidihedral group of order 32, shapes (1,3)x(2,5): high-order cubic components and binomial pairs attached to nonzero opposite-row cells should produce five fresh measured readings above eight and improve the cell-0.350 fitness record of 23.5.")
GROUP = ("L3 jump from the parent's order-40 C5 semidirect C8 group to an order-32 semidihedral presentation. Related order-32 (1,3)x(2,5) beds reached fitness 23.5, whereas the parent repeatedly measured [520,252,5]. Order-16 rotations and conjugation x->x^7 motivate new component searches; no group-distance advantage or untried-bed identity is assumed.")
SHAPES = ("Target cell 0.350 with A=(1,3), B=(2,5): n=544, kb=192, guaranteed rate 6/17, and component lengths/check counts 96/32 and 160/64. The window permits q<=35; this group fixes q=32. Parent minima occupied sector-one Acol/Bcol blocks, particularly Acol1/Bcol2. Changing B to two rows removes the inherited three-binomial single-row architecture. Verified complete ranks require 0.325<=k/n<0.375.")
FILLING = ("Twenty structural policies cross five A rotation-pair-order profiles, two B count architectures, and order-16/order-8 B binomial differences. All A cells are cubic; every B column has three terms and every binomial is attached to a nonzero opposite-row cell. HX=12 and HZ<=12. Compile component atoms solely from verified lifts, exactly exclude component words through four, census weight-six existence, and optimize actual kernel witnesses with equal deeper finalist budgets. Verify embeddings and stabilizer dressings on the complete CSS matrices, exclude full logicals through four, gate exact rates, and reject equal uncoloured Tanner invariants. Heuristic readings are capped at ten for selection and are never certificates. Historical order-40 helpers remain inactive; no parent reconstruction or archive fallback runs.")
FAMILY_NAME = "r035_SD32_13x25_attached_binomials_cubic_orders_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # SD32: x^16=y^2=1, y*x*y^-1=x^7; 7^2=1 mod 16.
    if not valid_q(q):
        raise ValueError("semidihedral SD32 requires order 32")
    els = [(a, b) for b in range(2) for a in range(16)]
    action = (1, 7)

    def red(g):
        return (g[0] % 16, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b] * c) % 16, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-action[b] * a) % 16, b)

    def idx(g):
        a, b = red(g)
        return a + 16 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _four_cycle_count(rows):
    """Four-cycles in one Tanner graph, using the verified bit rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _has_small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight 1..4.

    Enumerate supports of size at most two. Equal syndromes yield
    every commuting support of size at most four. Test differences
    against the opposite stabilizer row space, not merely the kernel.
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

    syndrome = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndrome[low.bit_length() - 1] ^= bit
            row ^= low

    seen = {0: 0}
    for i, s in enumerate(syndrome):
        v = 1 << i
        old = seen.get(s)
        if old is not None:
            if outside(v ^ old):
                return True
        else:
            seen[s] = v

    for i in range(n):
        vi = 1 << i
        si = syndrome[i]
        for j in range(i):
            s = si ^ syndrome[j]
            v = vi | (1 << j)
            old = seen.get(s)
            if old is not None:
                if outside(v ^ old):
                    return True
            else:
                seen[s] = v
    return False


@__import__("functools").lru_cache(maxsize=1)
def _shared_cokernel_bank(G):
    """Pure, cached optimization of fresh rank-aware design cards."""
    from collections import Counter

    q = G.q
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(3)]]
    dummy_b = [[[G.e] for _ in range(4)]]

    def component_rows(cells, side):
        if side == "A":
            hx, hz, n = lifted_matrices(G, cells, dummy_b)
            rows = []
            for row in hx[:q]:
                rows.append(sum(
                    ((row >> (4*j*q)) & mask) << (j*q)
                    for j in range(3)))
            return rows, 3*q
        hx, hz, n = lifted_matrices(G, dummy_a, cells)
        bm = (1 << (4*q)) - 1
        return [row & bm for row in hz[:q]], 4*q

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def dependency_space(rows):
        """Canonical exact space of dependencies between verified rows."""
        pivots, dependencies = {}, []
        for r, original in enumerate(rows):
            row, word = original, 1 << r
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (row, word)
                    break
                row ^= old[0]
                word ^= old[1]
            if not row:
                dependencies.append(word)

        basis = {}
        for word in dependencies:
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def fingerprint(hx, hz, n):
        """Permutation-invariant combined, uncoloured incidence signature.

        Different signatures imply different incidence graphs. Equal
        signatures are conservatively rejected, not asserted isomorphic.
        """
        rows = list(hx) + list(hz)
        adj = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adj[i][j] = overlap
                    adj[j][i] = overlap
        profiles = []
        for r, neighbors in enumerate(adj):
            items = list(neighbors.items())
            triangles = 0
            for a, (s, ws) in enumerate(items):
                for t, wt in items[:a]:
                    triangles += 2*ws*wt*adj[s].get(t, 0)
            profiles.append((
                rows[r].bit_count(),
                tuple(sorted(Counter(neighbors.values()).items())),
                sum(w*w for w in neighbors.values()),
                triangles))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(degrees).items())))

    # A single-block matrix is extracted from a full, verified LP lift.
    # Right translations merely permute its columns and are removed by
    # the column-incidence signature before blocks enter a class.
    classes = {}
    gauges = ([G.e] + [(a, 0) for a in range(1, 5)]
              + [(0, b) for b in range(1, 8)])
    for gauge in gauges:
        for a in range(1, q):
            for b in range(a + 1, q):
                cell = tuple(G.mul(gauge, g)
                             for g in (G.e, G.elements[a], G.elements[b]))
                A = [[list(cell), [G.e], [G.e]]]
                hx, hz, n = lifted_matrices(G, A, dummy_b)
                rows = [row & mask for row in hx[:q]]
                deficit = q - gf2_rank(rows)
                if deficit not in (4, 8):
                    continue
                if _has_small_logical(rows, [], q):
                    continue
                space = dependency_space(rows)
                incidence = tuple(sorted(columns(rows, q)))
                bucket = classes.setdefault(space, {})
                if incidence not in bucket and len(bucket) < 32:
                    bucket[incidence] = cell
        if sum(len(bucket) >= 3 for bucket in classes.values()) >= 24:
            break

    library = tuple(
        (space, tuple(bucket.values()))
        for space, bucket in classes.items() if len(bucket) >= 3)
    library = tuple(sorted(
        library, key=lambda item: (len(item[0]), item[0])))

    # These quotient models differ intrinsically from the parent's
    # preferred identity-plus-two-distinct-nonzero-quotient templates.
    b_models = ((0, 0, 1), (0, 0, 3), (0, 2, 6), (0, 1, 5))
    target_deficits = (4, 8, 4, 8, 4)
    target_spreads = (6, 6, 9, 9, 3)
    fingerprints, bank = set(), []

    def generators(rng):
        u = rng.randrange(5)
        v = rng.choice([a for a in range(5) if a != 2*u % 5])
        return ((u, 1), (v, 3), (rng.randrange(1, 5), 0))

    def triple(rng, model):
        used, cell = set(), []
        for b in model:
            choices = [(a, b) for a in range(5) if (a, b) not in used]
            g = rng.choice(choices)
            used.add(g)
            cell.append(g)
        base = G.pick(rng)
        return [G.mul(base, g) for g in cell]

    def draw_b(rng, mode, fallback):
        gs = generators(rng)
        if fallback:
            return [[[G.e, gs[0]], [G.e, gs[1]],
                     triple(rng, b_models[mode]),
                     triple(rng, b_models[(mode + 1) % 4])]]
        return [[[G.e, g] for g in gs]
                + [triple(rng, b_models[mode])]]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    for slot in range(N_SLOTS):
        card = slot % 20
        policy, mode = divmod(card, 4)
        rng = random.Random(40013573 + 100003*slot)
        winner = None

        # Rank deficit and quotient diversity are prescribed policies;
        # independent labels are optimized inside each policy.
        eligible = [
            item for item in library
            if len(item[0]) == target_deficits[policy]]
        if not eligible:
            eligible = list(library)

        for fallback in (False, True):
            if not fallback and not eligible:
                continue
            sa = [[2, 2, 2]] if fallback else [[3, 3, 3]]
            sb = [[2, 2, 3, 3]] if fallback else [[2, 2, 2, 3]]
            assert _grid_weights(sa, sb) <= 12
            assert _no_dead_lines(sa, sb)

            apool, aseen = [], set()
            for attempt in range(160):
                if fallback:
                    A = [[[G.e, g] for g in generators(rng)]]
                else:
                    space, cells = rng.choice(eligible)
                    A = [[list(cell) for cell in rng.sample(cells, 3)]]
                sig = signature(A, dummy_b)
                if sig in aseen:
                    continue
                aseen.add(sig)
                ar, width = component_rows(A, "A")
                if _has_small_logical(ar, [], width):
                    continue
                deficit = q - gf2_rank(ar)
                spread = sum(len({g[1] for g in cell}) for cell in A[0])
                key = (deficit,
                       -abs(spread - target_spreads[policy]),
                       -_four_cycle_count(ar))
                apool.append((key, A))
                apool.sort(key=lambda item: item[0], reverse=True)
                del apool[6:]
                if len(apool) >= 6 and attempt >= 48:
                    break
            if not apool:
                continue

            bpool, bseen = [], set()
            for attempt in range(128):
                B = draw_b(rng, mode, fallback)
                sig = signature(dummy_a, B)
                if sig in bseen:
                    continue
                bseen.add(sig)
                br, width = component_rows(B, "B")
                if _has_small_logical(br, [], width):
                    continue
                key = (_four_cycle_count(br), -gf2_rank(br))
                bpool.append((key, B))
                bpool.sort(key=lambda item: item[0])
                del bpool[6:]
                if len(bpool) >= 6 and attempt >= 48:
                    break
            if not bpool:
                continue

            proposals = []
            for ak, A in apool:
                for bk, B in bpool:
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (240 <= k and 1000*k < 525*n):
                        continue
                    cycles = -ak[2] + bk[0]
                    # Five is a conservative optimization target, not
                    # a newly certified or sampled distance reading.
                    key = (k, -cycles, ak[1])
                    proposals.append((key, A, B, hx, hz, n))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, A, B, hx, hz, n in proposals[:12]:
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in fingerprints:
                    continue
                assert css_ok(hx, hz), "verified lift failed CSS commutation"
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert _grid_weights(
                    [[len(cell) for cell in row] for row in A],
                    [[len(cell) for cell in row] for row in B]) <= 12
                winner = (A, B, k, fp, fallback)
                break
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B, k, fp, fallback = winner
        fingerprints.add(fp)
        bank.append({
            "cells": {"A": A, "B": B}, "k": k,
            "card": card, "fallback": fallback,
            "screen_no_le4": True})

    return tuple(bank)


@__import__("functools").lru_cache(maxsize=4)
def _legacy_rankbudget_ctx(rng, G, idx):
    """Inactive order-40 historical optimizer; never called by this program."""
    if idx >= 0:
        selected = _legacy_rankbudget_ctx(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh half-rate rank-budget design passed exact screening")
        return selected

    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 40
    mask = (1 << q) - 1
    widths = {"A": 3, "B": 4}

    def columns(rows, width):
        out = [0] * width
        for r, original in enumerate(rows):
            word = original
            while word:
                bit = word & -word
                out[bit.bit_length() - 1] |= 1 << r
                word ^= bit
        return tuple(out)

    def basis(words):
        out = {}
        for word in words:
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def outside(word, span):
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def canonical(words):
        span = basis(words)
        pivots = sorted(span)
        for p in pivots:
            for h in pivots:
                if h > p and ((span[h] >> p) & 1):
                    span[h] ^= span[p]
        return tuple(span[p] for p in pivots)

    def component_rows(cols):
        rows = [0] * q
        for c, original in enumerate(cols):
            word = original
            while word:
                bit = word & -word
                rows[bit.bit_length() - 1] |= 1 << c
                word ^= bit
        return rows

    def short_module(cols):
        """Exact span generated by all kernel supports of size <=4.

        One representative per syndrome suffices to generate every
        collision difference. This is a span census, not a distance probe.
        """
        seen, span = {0: 0}, {}
        bits = [1 << i for i in range(len(cols))]

        def observe(syndrome, word):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
                return
            word ^= old
            while word:
                p = word.bit_length() - 1
                previous = span.get(p)
                if previous is None:
                    span[p] = word
                    break
                word ^= previous

        for i, value in enumerate(cols):
            observe(value, bits[i])
        for i, value in enumerate(cols):
            for j in range(i):
                observe(value ^ cols[j], bits[i] | bits[j])
        return tuple(span[p] for p in sorted(span))

    # Compile every placement from complete, verified unit-cell lifts.
    # Matrix assembly below is F2-linear superposition, not a hand lift.
    actions, atlas = {}, {}
    for side, width in widths.items():
        for j in range(width):
            units, blocks = [], []
            for g in G.elements:
                A = [[[] for _ in range(3)]]
                B = [[[] for _ in range(4)]]
                (A if side == "A" else B)[0][j] = [g]
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 520
                units.append((
                    tuple((r, v) for r, v in enumerate(hx) if v),
                    tuple((r, v) for r, v in enumerate(hz) if v)))
                if j == 0:
                    source = hx if side == "A" else hz
                    blocks.append(columns(
                        [v & mask for v in source[:q]], q))
            actions[side, j] = tuple(units)
            if j == 0:
                atlas[side] = tuple(blocks)

    def block(side, cell):
        out = [0] * q
        for g in cell:
            for h, value in enumerate(atlas[side][G.idx(g)]):
                out[h] ^= value
        return tuple(out)

    def full_lift(A, B):
        hx, hz = [0] * (4*q), [0] * (3*q)
        for side, grid in (("A", A), ("B", B)):
            for j, cell in enumerate(grid[0]):
                for g in cell:
                    xs, zs = actions[side, j][G.idx(g)]
                    for r, value in xs:
                        hx[r] ^= value
                    for r, value in zs:
                        hz[r] ^= value
        return hx, hz, 520

    # Exhaust normalized supports, including quartics. Sorted column
    # incidences remove single-block column-gauge copies, not kernels.
    others = [g for g in G.elements if G._key(g) != G._key(G.e)]
    libraries, targets = {}, {}
    for side in ("A", "B"):
        libraries[side] = {}
        target_set = set()
        for size in (2, 3, 4):
            records = {}
            for tail in combinations(others, size - 1):
                cell = (G.e,) + tail
                cols = block(side, cell)
                space = canonical(cols)
                incidence = tuple(sorted(cols))
                records.setdefault(incidence, dict(
                    cell=cell, cols=cols, space=space, rank=len(space),
                    labels=tuple(G.idx(g) for g in cell),
                    spread=len({g[1] for g in cell})))
                if size == 3 and 30 <= len(space) <= 36:
                    target_set.add(space)
            libraries[side][size] = tuple(records.values())

        # Joined binomial images include deficit-one/two architectures
        # absent from a library indexed only by singular cubic images.
        bins = libraries[side][2]
        for i, first in enumerate(bins):
            target_set.add(first["space"])
            for second in bins[:i]:
                joined = canonical(first["space"] + second["space"])
                if 30 <= len(joined) < q:
                    target_set.add(joined)
        targets[side] = tuple(sorted(
            target_set, key=lambda s: (-len(s), s)))

    containment_cache, component_cache = {}, {}

    def contained(side, space, size):
        key = side, space, size
        if key not in containment_cache:
            span = basis(space)
            containment_cache[key] = tuple(
                record for record in libraries[side][size]
                if record["rank"] <= len(space)
                and all(not outside(v, span) for v in record["space"]))
        return containment_cache[key]

    def component_pool(side, counts, wanted, mode, local, batch):
        spaces = list(targets[side])
        local.shuffle(spaces)
        spaces.sort(key=lambda s: abs(q - len(s) - wanted))
        groups = []
        multiplicities = Counter(counts)
        for space in spaces[:48]:
            choices = {
                size: contained(side, space, size)
                for size in multiplicities}
            if all(len(choices[size]) >= amount
                   for size, amount in multiplicities.items()):
                groups.append(choices)

        strata, seen = {}, set()
        for attempt in range(192 + 64*batch):
            # A quarter of proposals are unrestricted image joins.
            # Common-image containment is a proposal tool, never a gate.
            choices = (
                local.choice(groups) if groups and attempt % 4
                else {size: libraries[side][size]
                      for size in multiplicities})
            selected = []
            for size, amount in sorted(multiplicities.items()):
                selected.extend(local.sample(choices[size], amount))
            selected.sort(key=lambda r: (len(r["cell"]), r["labels"]))
            labels = tuple(r["labels"] for r in selected)
            if labels in seen:
                continue
            seen.add(labels)
            cache_key = side, labels
            item = component_cache.get(cache_key)
            if item is None:
                cols = tuple(v for record in selected for v in record["cols"])
                deficit = q - gf2_rank(cols)
                if not 0 <= deficit <= 10:
                    continue
                module = short_module(cols)
                item = dict(
                    grid=[[list(r["cell"]) for r in selected]],
                    cols=cols, deficit=deficit, module=module,
                    cycles=_four_cycle_count(component_rows(cols)),
                    spread=sum(r["spread"] for r in selected),
                    labels=labels)
                component_cache[cache_key] = item

            # Preserve dirty and clean module strata separately. In
            # particular, a low-cycle clean stratum cannot erase every
            # high-rank architecture before quantum protection is tested.
            stratum = item["deficit"], min(len(item["module"]), 8)
            pool = strata.setdefault(stratum, [])
            pool.append(item)
            spread_target = (
                len(counts)*2 if mode == 1 else sum(counts))
            pool.sort(key=lambda a: (
                len(a["module"]),
                abs(a["spread"] - spread_target),
                a["cycles"], a["labels"]))
            del pool[4:]

        answer = [a for pool in strata.values() for a in pool]
        local.shuffle(answer)
        answer.sort(key=lambda a: (
            abs(a["deficit"] - wanted),
            len(a["module"]), a["cycles"]))
        # Include one representative from every retained rank/module
        # stratum before filling the remaining shortlist positions.
        leaders = [pool[0] for pool in strata.values()]
        leaders.sort(key=lambda a: (
            len(a["module"]), abs(a["deficit"] - wanted), a["cycles"]))
        result = leaders[:16]
        used = {a["labels"] for a in result}
        result.extend(a for a in answer if a["labels"] not in used)
        return result[:24]

    def protected(a, b, xs, zs):
        for seed in a["module"]:
            for beta in range(4):
                word = sum(
                    ((seed >> (j*q)) & mask) << ((4*j+beta)*q)
                    for j in range(3))
                if outside(word, zs):
                    return False
        for seed in b["module"]:
            for j in range(3):
                if outside(seed << (4*j*q), xs):
                    return False
        return True

    def logical_five(checks, stabilizers, n):
        """Exact <=5 existence AFTER complete logical-4 exclusion.

        Equal-syndrome tails of size <=2 differ by a stabilizer after
        that screen. Consequently one tail representative suffices.
        Every remaining support must touch an unsatisfied check; the
        branching is exhaustive and has no time or node cutoff.
        """
        cols = columns(checks, n)
        span = basis(stabilizers)
        bits = [1 << i for i in range(n)]
        incidence = [[] for _ in checks]
        tails = {0: 0}
        for i, value in enumerate(cols):
            tails.setdefault(value, bits[i])
            pending = value
            while pending:
                bit = pending & -pending
                incidence[bit.bit_length() - 1].append(i)
                pending ^= bit
        for i, value in enumerate(cols):
            for j in range(i):
                tails.setdefault(value ^ cols[j], bits[i] | bits[j])
        max_degree = max(v.bit_count() for v in cols)

        def visit(value, remaining, word):
            if not value:
                return outside(word, span)
            if value.bit_count() > max_degree*remaining:
                return False
            if remaining <= 2:
                tail = tails.get(value)
                return (tail is not None
                        and tail.bit_count() <= remaining
                        and outside(word ^ tail, span))
            check = (value & -value).bit_length() - 1
            for j in incidence[check]:
                if not word & bits[j]:
                    if visit(value ^ cols[j], remaining - 1,
                             word ^ bits[j]):
                        return True
            return False

        return any(visit(value, 4, bits[i])
                   for i, value in enumerate(cols))

    def fingerprint(hx, hz, n):
        """Uncoloured incidence invariant; inequality proves nonisomorphism."""
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        adjacency = [0] * len(rows)
        overlaps = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                size = (row & rows[j]).bit_count()
                if size:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    overlaps[i][j] = overlaps[j][i] = size
        keys = []
        for i, neighbors in enumerate(overlaps):
            profile = Counter(
                (weights[j], size,
                 (adjacency[i] & adjacency[j]).bit_count())
                for j, size in neighbors.items())
            keys.append((weights[i], tuple(sorted(profile.items()))))
        history = []
        for _ in range(3):
            histogram = Counter(keys)
            history.append(tuple(sorted(histogram.items())))
            palette = {key: t for t, key in enumerate(sorted(histogram))}
            colours = [palette[key] for key in keys]
            keys = [
                (colours[i], tuple(sorted(Counter(
                    (colours[j], size)
                    for j, size in neighbors.items()).items())))
                for i, neighbors in enumerate(overlaps)]
        incidents = [[] for _ in range(n)]
        for r, original in enumerate(rows):
            word = original
            while word:
                bit = word & -word
                incidents[bit.bit_length() - 1].append(colours[r])
                word ^= bit
        return n, tuple(history), tuple(sorted(Counter(
            tuple(sorted(v)) for v in incidents).items()))

    # NEGATIVE ARCHIVE ONLY. These reconstructed cells are never
    # proposals, initial candidates, reserves, or emitted outputs.
    emitted = set()
    for old in _shared_cokernel_bank(G):
        if old is not None:
            hx, hz, n = lifted_matrices(
                G, old["cells"]["A"], old["cells"]["B"])
            emitted.add(fingerprint(hx, hz, n))

    primary = (
        ((3,3,4), (2,2,2,2)),
        ((2,4,4), (2,2,2,2)),
        ((2,3,4), (2,2,2,2)),
        ((2,2,2), (2,2,3,3)),
        ((2,2,2), (2,2,2,4)),
    )
    reserves = (
        ((2,3,3), (2,2,2,3)),
        ((3,3,3), (2,2,2,3)),
    )
    wanted_deficits = (4, 8, 2, 1)
    bank = []

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        local = random.Random(40078127 + 104729*slot)
        winner = None
        plans = (primary[architecture],) + reserves

        for stage, (sa, sb) in enumerate(plans):
            assert _grid_weights([list(sa)], [list(sb)]) <= W_MAX
            assert _no_dead_lines([list(sa)], [list(sb)])
            ap = component_pool(
                "A", sa, wanted_deficits[mode], mode, local, stage)
            # Rank surplus on B is important for the all-binomial A
            # architectures; do not always optimize B towards full rank.
            b_wanted = (
                4 if max(sa) == 2 else (1, 4, 2, 0)[mode])
            bp = component_pool(
                "B", sb, b_wanted, mode, local, stage)
            proposals = [(a, b) for a in ap for b in bp]
            local.shuffle(proposals)
            proposals.sort(key=lambda pair: (
                -(3*pair[0]["deficit"] + 2*pair[1]["deficit"]),
                len(pair[0]["module"]) + len(pair[1]["module"]),
                pair[0]["cycles"] + pair[1]["cycles"]))

            finalists = []
            for a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = full_lift(A, B)
                xs, zs = basis(hx), basis(hz)
                k = n - len(xs) - len(zs)
                if not (475*n <= 1000*k < 525*n):
                    continue
                if not protected(a, b, xs, zs):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue
                finalists.append((A, B, hx, hz, n, k, fp, a, b))
                if len(finalists) >= 6:
                    break

            for A, B, hx, hz, n, k, fp, a, b in finalists:
                floor = 5
                if (not logical_five(hz, hx, n)
                        and not logical_five(hx, hz, n)):
                    floor = 6
                objective = (
                    k*floor*floor, k, -stage,
                    -a["cycles"] - b["cycles"])
                if winner is None or objective > winner[0]:
                    winner = (
                        objective, A, B, hx, hz, k, fp, floor, stage)
            # A better rank at the same exact exclusion floor is useful.
            # Otherwise inspect new assemblies in the reserve profiles.
            if winner is not None:
                if winner[7] >= 6 or winner[5] > 252 or stage == 2:
                    break

        if winner is None:
            bank.append(None)
            continue
        objective, A, B, hx, hz, k, fp, floor, stage = winner
        vx, vz, n = lifted_matrices(G, A, B)
        assert hx == vx and hz == vz and n == 520
        assert k == lifted_k(G, A, B)
        assert css_ok(hx, hz)
        assert max(r.bit_count() for r in hx + hz) <= W_MAX
        assert _grid_weights(
            [[len(c) for c in row] for row in A],
            [[len(c) for c in row] for row in B]) <= W_MAX
        assert all(0 < len(c) == len(set(c)) <= 5
                   for grid in (A, B) for row in grid for c in row)
        emitted.add(fp)
        bank.append(dict(
            cells={
                side: tuple(tuple(tuple(c) for c in row) for row in grid)
                for side, grid in (("A", A), ("B", B))},
            k=k, card=(architecture, mode), stage=stage,
            verified_exclusion_floor=floor, selection=objective))
    return tuple(bank)


def _sd_columns(rows, width):
    cols = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return tuple(cols)


def _sd_basis(words):
    span = {}
    for original in words:
        word = original
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return span


def _sd_outside(word, span):
    while word:
        p = word.bit_length() - 1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _sd_clean4(cols):
    """Exact component-kernel exclusion through four."""
    seen = {0}
    for value in cols:
        if value in seen:
            return False
        seen.add(value)
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pair = value ^ other
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _sd_six_hit(cols):
    """Exact weight-six existence after clean4, for odd-weight columns.

    Odd columns exclude odd-weight kernel words. Distinct colliding
    triples must therefore be disjoint and give a weight-six support.
    """
    assert all(value.bit_count() & 1 for value in cols)
    seen = set()
    for i, value in enumerate(cols):
        for j in range(i):
            pair = value ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
    return False


def _sd_probe(cols, local, trials, stabilizers=None, seeds=()):
    """Actual kernel/logical witnesses; a high reading is not a proof."""
    n = len(cols)
    span = None if stabilizers is None else _sd_basis(stabilizers)
    generators = () if stabilizers is None else tuple(stabilizers)
    order = list(range(n))
    best = n + 1

    def syndrome(word):
        result = 0
        while word:
            bit = word & -word
            result ^= cols[bit.bit_length() - 1]
            word ^= bit
        return result

    memory = {word for word in seeds if word and syndrome(word) == 0}

    def admissible(word):
        return word and (span is None or _sd_outside(word, span))

    def observe(word):
        nonlocal best
        if admissible(word):
            best = min(best, word.bit_count())

    for word in memory:
        observe(word)

    for _ in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            value, word = cols[c], 1 << c
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                dependencies.append(word)
                observe(word)

        donors = sorted(set(dependencies) | memory,
                        key=lambda w: (w.bit_count(), w))[:40]
        logical = sorted(
            (w for w in set(dependencies) | memory if admissible(w)),
            key=lambda w: (w.bit_count(), w))[:16]
        discoveries = set(donors)
        for seed in logical:
            word = seed
            for sweep in range(3):
                previous = word.bit_count()
                for direction in donors:
                    trial = word ^ direction
                    if (trial.bit_count() < word.bit_count()
                            and admissible(trial)):
                        word = trial
                for direction in generators:
                    trial = word ^ direction
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word.bit_count() == previous:
                    break
            observe(word)
            discoveries.add(word)

        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                observe(trial)
                if trial and trial.bit_count() <= min(best + 4, 18):
                    discoveries.add(trial)
        # Preserve logical directions separately from short stabilizers.
        good = sorted((w for w in discoveries if admissible(w)),
                      key=lambda w: (w.bit_count(), w))[:24]
        raw = sorted(discoveries, key=lambda w: (w.bit_count(), w))[:24]
        memory = set(good + raw)
        if best <= 4:
            break
    return best, tuple(sorted(memory, key=lambda w: (w.bit_count(), w)))


def _sd_fingerprint(hx, hz, n):
    """Uncoloured invariant: unequal values certify nonisomorphism."""
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    overlap = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlap[i][j] = overlap[j][i] = size
    keys = [
        (weights[i], tuple(sorted(Counter(
            (weights[j], size,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, size in neighbors.items()).items())))
        for i, neighbors in enumerate(overlap)]
    history = []
    for _ in range(3):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: c for c, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], size)
                for j, size in neighbors.items()).items())))
            for i, neighbors in enumerate(overlap)]
    incidents = [[] for _ in range(n)]
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            incidents[bit.bit_length() - 1].append(colours[r])
            row ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(v)) for v in incidents).items())))


@__import__("functools").lru_cache(maxsize=1)
def _sd_bank(G):
    """Fresh structural cards; all parity actions originate in verified lifts."""
    from math import gcd
    assert G.q == 32
    q, mask = G.q, (1 << G.q) - 1
    zero_a = [[[], [], []]]
    zero_b = [[[] for _ in range(5)] for _ in range(2)]
    atlas = {}

    for side in ("A", "B"):
        atoms = []
        for g in G.elements:
            A = [[[g], [], []]] if side == "A" else zero_a
            B = ([[[g]] + [[] for _ in range(4)],
                  [[] for _ in range(5)]]
                 if side == "B" else zero_b)
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 544
            source = hx if side == "A" else hz
            atoms.append(_sd_columns(
                [row & mask for row in source[:q]], q))
        atlas[side] = tuple(atoms)

    def compile_component(side, grid):
        width = 3 if side == "A" else 5
        cols = [0] * (width * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for h, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+h] ^= value << (i*q)
        return tuple(cols)

    def component_rows(cols, height):
        return _sd_columns(cols, height)

    steps = {
        order: tuple(d for d in range(1, 16)
                     if 16 // gcd(d, 16) == order)
        for order in (16, 8, 4)
    }
    a_cards = (
        (16, 16, 16), (16, 16, 8), (16, 8, 8),
        (8, 8, 8), (16, 16, 4),
    )

    def cubic(local, order):
        base = G.pick(local)
        support = (G.e, (local.choice(steps[order]), 0),
                   (local.randrange(16), 1))
        return [G.mul(base, g) for g in support]

    def draw_a(local, orders):
        used, row = set(), []
        for order in orders:
            winner = None
            for _ in range(24):
                cell = cubic(local, order)
                differences = [
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h]
                unique = set(differences)
                cost = len(differences) - len(unique) + len(unique & used)
                if winner is None or cost < winner[0]:
                    winner = cost, cell, unique
                if cost == 0:
                    break
            _, cell, unique = winner
            row.append(cell)
            used.update(unique)
        return [row]

    def draw_b(local, architecture, order, phase):
        isolated = 1 + architecture
        grid = [[[] for _ in range(5)] for _ in range(2)]
        for j in range(5):
            if j < isolated:
                grid[0][j] = cubic(
                    local, (16, 8)[(j + phase) & 1])
            else:
                grid[0][j] = [G.pick(local)]
                base = G.pick(local)
                delta = (local.choice(steps[order]), 0)
                grid[1][j] = [base, G.mul(base, delta)]
        return grid

    def component_pool(local, side, orders, architecture, order, phase):
        raw, seen = [], set()
        quota = 192 if side == "A" else 256
        for attempt in range(quota):
            grid = (draw_a(local, orders) if side == "A" else
                    draw_b(local, architecture, order, phase))
            labels = tuple(tuple(sorted(G.idx(g) for g in cell))
                           for row in grid for cell in row)
            if labels in seen:
                continue
            seen.add(labels)
            cols = compile_component(side, grid)
            if not _sd_clean4(cols):
                continue
            rank = gf2_rank(cols)
            reading, seeds = _sd_probe(cols, local, 4)
            rows = component_rows(cols, (1 if side == "A" else 2)*q)
            cycles = _four_cycle_count(rows)
            raw.append(dict(
                grid=grid, cols=cols, rank=rank, d=reading,
                seeds=seeds, cycles=cycles, labels=labels))
            raw.sort(key=lambda a: (
                min(a["d"], 12), -a["rank"], -a["cycles"]),
                reverse=True)
            del raw[8:]

        for item in raw:
            # This census is exact; the subsequent witness reading is not.
            hit6 = _sd_six_hit(item["cols"])
            if hit6:
                item["d"] = min(item["d"], 6)
            item["d"], item["seeds"] = _sd_probe(
                item["cols"], local, 40, seeds=item["seeds"])
            if hit6:
                item["d"] = min(item["d"], 6)
            item["clean_through7"] = not hit6
        raw.sort(key=lambda a: (
            min(a["d"], 10), a["clean_through7"],
            -a["rank"], -a["cycles"]), reverse=True)
        return raw[:3]

    def embedded_seeds(a, b, hx, hz, n):
        xs, zs = _sd_basis(hx), _sd_basis(hz)
        xc, zc = _sd_columns(hz, n), _sd_columns(hx, n)

        def retain(word, cols, span, destination):
            syndrome, pending = 0, word
            while pending:
                bit = pending & -pending
                syndrome ^= cols[bit.bit_length() - 1]
                pending ^= bit
            if not syndrome and _sd_outside(word, span):
                destination.append(word)

        xwords, zwords = [], []
        for seed in b["seeds"][:16]:
            for j in range(3):
                retain(seed << (j*5*q), xc, xs, xwords)
        for seed in a["seeds"][:16]:
            for beta in range(5):
                word = sum(
                    ((seed >> (j*q)) & mask) << ((5*j+beta)*q)
                    for j in range(3))
                retain(word, zc, zs, zwords)
        return xc, zc, tuple(xwords), tuple(zwords)

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        model, remainder = divmod(slot % 20, 4)
        architecture, mode = divmod(remainder, 2)
        order = (16, 8)[mode]
        local = random.Random(32035073 + 104729*slot)
        orders = a_cards[model]
        isolated = 1 + architecture
        sa = [[3, 3, 3]]
        sb = [[3]*isolated + [1]*(5-isolated),
              [0]*isolated + [2]*(5-isolated)]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        winner = None

        for retry in range(2):
            ap = component_pool(
                local, "A", orders, architecture, order, slot//20+retry)
            bp = component_pool(
                local, "B", orders, architecture, order, slot//20+retry)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (325*n <= 1000*k < 375*n):
                        continue
                    fp = _sd_fingerprint(hx, hz, n)
                    if fp in emitted:
                        continue
                    d = min(a["d"], b["d"], 10)
                    proposals.append((
                        (k*d*d, d, -a["cycles"]-b["cycles"]),
                        a, b, hx, hz, n, k, fp))
            proposals.sort(key=lambda p: p[0], reverse=True)

            finalists = []
            for _, a, b, hx, hz, n, k, fp in proposals:
                if (_has_small_logical(hx, hz, n)
                        or _has_small_logical(hz, hx, n)):
                    continue
                xc, zc, xseeds, zseeds = embedded_seeds(
                    a, b, hx, hz, n)
                dx, xseeds = _sd_probe(
                    xc, local, 12, hx, xseeds)
                dz, zseeds = _sd_probe(
                    zc, local, 12, hz, zseeds)
                finalists.append(dict(
                    A=a["grid"], B=b["grid"], hx=hx, hz=hz,
                    n=n, k=k, fp=fp, xc=xc, zc=zc,
                    xseeds=xseeds, zseeds=zseeds, dx=dx, dz=dz,
                    da=a["d"], db=b["d"],
                    cycles=a["cycles"]+b["cycles"]))
                if len(finalists) >= 3:
                    break

            for item in finalists:
                dx, _ = _sd_probe(
                    item["xc"], local, 32, item["hx"], item["xseeds"])
                dz, _ = _sd_probe(
                    item["zc"], local, 32, item["hz"], item["zseeds"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                d = min(item["da"], item["db"],
                        item["dx"], item["dz"], 10)
                item["merit"] = (
                    item["k"]*d*d, d, item["k"], -item["cycles"])
                if winner is None or item["merit"] > winner["merit"]:
                    winner = item
            if winner is not None and winner["merit"][1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert winner["n"] == 544
        assert winner["k"] == lifted_k(G, A, B) > 0
        assert css_ok(winner["hx"], winner["hz"])
        assert max(r.bit_count()
                   for r in winner["hx"] + winner["hz"]) <= 12
        assert all(len(cell) == len(set(cell)) <= 5
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={
                side: tuple(tuple(tuple(cell) for cell in row)
                            for row in grid)
                for side, grid in (("A", A), ("B", B))},
            k=winner["k"], card=(model, architecture, order),
            heuristic_selection=winner["merit"],
            screen_no_logical_le4=True))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's fresh semidihedral filling."""
    selected = _sd_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh semidihedral filling passed target-rate screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel rank selection and component four-cycle preferences remove the sector-one weight-five bottleneck; all five measured current leaders remained k=252, d30s=5, with thousands of weight-five witnesses.
# REFUTED: Quartic rank-budget architectures improve this order-40 half-rate bed; the measured program's five leaders instead retained cubic/binomial reserve supports, k=252, d30s=5 and fitness 12.1150.
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
