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
CONCEPT = ("Hypothesis: C13:C6(s=4), order 78, A=(1,3), B=(2,5), with fully cubic component columns and witness-directed voltage repair produces five selected distance readings >=12 in cell 0.350. This is a measurement hypothesis, not a quantum-distance certificate.")
GROUP = ("Jump to the faithful split-metacyclic group C13:C6(s=4): 4^3=-1 and 4^6=1 modulo 13. Its order 78 differs from the parent's 110, and this group/shape bed is absent from BEDS. The related nonabelian order-120 (1,3)x(1,3) bed reached readings twelve; this motivates cubic components without establishing transfer. Order 78 gives n=1326.")
SHAPES = ("A=(1,3), B=(2,5): factor seventeen gives n=1326, k>=78*2*3=468 and guaranteed rate 6/17=0.352941 in the comparatively weak 0.350 frontier cell. Components have 234 and 390 variables, with 78 and 156 checks. Parent minima were entirely sector one, particularly Acol1 across Bcol1..3. An additional B check row permits all B columns to be cubic instead of retaining the parent's degree-two bottleneck. Exact full ranks gate [0.325,0.375).")
FILLING = ("Twenty designs cross five cubic-column concentration profiles with four relative-voltage models. A has three trinomials and row total nine; B has five cubic columns and row totals seven/eight. Hence HX=9+3=12 and HZ<=8+3=11. Whole-cell directed-difference exclusions, verified component cycle optimization, exact kernel-four exclusion, bounded parity-completion searches and independent elimination challenges select fillings. Actual witnessed blocks direct whole-column redraws, with historical witnesses checked before reuse. High readings and budget stops never certify distance; final original-group matrices verify ranks, weights and CSS.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Frob13x6_13x25_cubic_witness10_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 78

def custom_ops(q):
    # (a,b) acts on F13 as t -> 4^b*t+a.
    # The faithful action has order six.
    if not valid_q(q):
        raise ValueError("C13:C6 requires order 78")
    powers = tuple(pow(4, b, 13) for b in range(6))
    els = [(a, b) for b in range(6) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 13, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi] * a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Column-major B counts: five inequivalent concentration profiles.
# Each column has degree three; row totals are exactly seven/eight.
_CUBIC_B = (
    ((2, 1), (2, 1), (1, 2), (1, 2), (1, 2)),
    ((3, 0), (1, 2), (1, 2), (1, 2), (1, 2)),
    ((3, 0), (0, 3), (2, 1), (1, 2), (1, 2)),
    ((3, 0), (0, 3), (0, 3), (2, 1), (2, 1)),
    ((3, 0), (3, 0), (0, 3), (0, 3), (1, 2)),
)


def _bits(word):
    while word:
        low = word & -word
        yield low.bit_length() - 1
        word ^= low


def _cubic_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _cubic_cell(rng, G, count, mode):
    """Relative supports vary structurally; no orbit images are submitted."""
    if count == 0:
        return []
    root = G.pick(rng)
    if count == 1:
        return [root]
    if count == 2:
        if mode == 0:
            ratio = (rng.randrange(1, 13), 0)  # order thirteen
        else:
            gap = ((1, 5), (2, 4), (1, 2))[mode - 1]
            ratio = (rng.randrange(13), rng.choice(gap))
        return [root, G.mul(root, ratio)]

    assert count == 3
    for _ in range(256):
        if mode in (0, 3):
            u = (rng.randrange(1, 13), 0)
            v = (rng.randrange(13), 1 if mode == 0 else 2)
        else:
            u = (rng.randrange(13), 1)
            v = (rng.randrange(13), 2 if mode == 1 else 5)
        if G.mul(u, v) == G.mul(v, u):
            continue
        cell = [root, G.mul(root, u), G.mul(root, v)]
        # Both directed-difference orientations matter for nonabelian lifts.
        left = {G.idx(G.mul(G.inv(a), b))
                for a in cell for b in cell if a != b}
        right = {G.idx(G.mul(a, G.inv(b)))
                 for a in cell for b in cell if a != b}
        if len(left) == len(right) == 6:
            return cell
    raise ValueError("Sidon cubic cell could not be filled")


def _cubic_record(G, probe, grid, side, mode):
    """Every matrix action is obtained from the verified construction."""
    indexed = [[[G.idx(g) for g in cell] for cell in row]
               for row in grid]
    if side == "A":
        rows, _, _ = lifted_matrices(probe, indexed, [[[]]])
    else:
        _, rows, _ = lifted_matrices(probe, [[[]]], indexed)
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in rows]
    data = _component_data(rows, width)
    if data is None:
        return None
    columns, four, triangles = data
    assert all(col.bit_count() == 3 for col in columns)
    return {
        "grid": _cubic_copy(grid), "rows": rows, "columns": columns,
        "cost": 800 * four + (1, 2, 3, 5)[mode] * triangles,
        "cycles": (four, triangles), "words": set(), "d": width + 1,
        "signature": tuple(tuple(sorted(G.idx(g) for g in cell))
                           for row in grid for cell in row),
    }


def _cubic_syndrome(columns, word):
    result = 0
    for v in _bits(word):
        result ^= columns[v]
    return result


def _cubic_note(record, word):
    """Only retain nonzero, explicitly verified component-kernel words."""
    if not word:
        return
    weight = word.bit_count()
    if weight > record["d"] + 4:
        return
    assert _cubic_syndrome(record["columns"], word) == 0
    record["d"] = min(record["d"], weight)
    record["words"].add(word)
    if len(record["words"]) > 64:
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:48])


def _cubic_probe(rng, record, trials):
    """Independent information sets plus genuine kernel-basis descent."""
    columns = record["columns"]
    order = list(range(len(columns)))
    retained = []
    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
        for v in order:
            syndrome, word = columns[v], 1 << v
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if syndrome:
                continue
            basis.append(word)
        light = sorted(basis, key=int.bit_count)[:16]
        for word in light:
            _cubic_note(record, word)
            for other in basis:
                candidate = word ^ other
                if candidate and candidate.bit_count() <= record["d"]:
                    _cubic_note(record, candidate)
        rng.shuffle(basis)
        for source in light[:6]:
            word = source
            for sweep in range(2):
                changed = False
                for other in retained + basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() < word.bit_count():
                        word, changed = candidate, True
                if not changed:
                    break
            _cubic_note(record, word)
        retained = sorted(set(retained + light),
                          key=int.bit_count)[:24]
        if record["d"] <= 4:
            break


def _cubic_completion(rng, record, q, budget, cap=10):
    """Budgeted anchored parity completion; absence is never a certificate."""
    columns = record["columns"]
    width = len(columns)
    singles, pairs = {}, {}
    incidence = [0] * len(record["rows"])
    for v, col in enumerate(columns):
        bit = 1 << v
        singles.setdefault(col, bit)
        for r in _bits(col):
            incidence[r] |= bit
        for u in range(v):
            pairs.setdefault(col ^ columns[u], bit | (1 << u))
    roots = list(range(0, width, q))
    rng.shuffle(roots)
    priorities = [rng.random() for _ in columns]
    all_variables = (1 << width) - 1

    for root in roots:
        remaining = [max(1, budget // len(roots))]

        def visit(value, word, allowed):
            if remaining[0] <= 0:
                return
            remaining[0] -= 1
            left = min(cap, record["d"]) - word.bit_count()
            if not value:
                _cubic_note(record, word)
                return
            if left <= 0 or value.bit_count() > 3 * left:
                return
            one = singles.get(value, 0)
            if one and one & allowed == one:
                _cubic_note(record, word | one)
            pair = pairs.get(value, 0)
            if left >= 2 and pair and pair & allowed == pair:
                _cubic_note(record, word | pair)
            left = min(cap, record["d"]) - word.bit_count()
            if left <= 2:
                return

            chosen, smallest = 0, width + 1
            for r in _bits(value):
                available = incidence[r] & allowed
                count = available.bit_count()
                if not count:
                    return
                if count < smallest:
                    chosen, smallest = available, count
            choices = list(_bits(chosen))
            choices.sort(key=lambda v: (
                -(columns[v] & value).bit_count(), priorities[v]))
            live = allowed
            for v in choices:
                bit = 1 << v
                live &= ~bit
                visit(value ^ columns[v], word | bit, live)
                if remaining[0] <= 0:
                    break

        # The regular component symmetry permits an identity anchor in
        # the first occupied block. We do not construct symmetry-image codes.
        allowed = all_variables & ~((1 << (root + 1)) - 1)
        visit(columns[root], 1 << root, allowed)


def _cubic_merit(record):
    minimum = record["d"]
    multiplicity = sum(w.bit_count() == minimum
                       for w in record["words"])
    # Fourteen is a selection ceiling, not an exclusion floor.
    return (min(14, minimum), -multiplicity, -record["cost"])


def _cubic_pool(rng, G, probe, support, side, mode):
    """Cycle-safe multistarts followed by actual short-word column repair."""
    import math
    width = len(support[0])
    side_offset = int(side == "B")
    keepers = {}

    def draw(i, j):
        prescription = (mode + j + i + side_offset) % 4
        return _cubic_cell(rng, G, support[i][j], prescription)

    def fresh():
        return [[draw(i, j) for j in range(width)]
                for i in range(len(support))]

    def inspect(grid):
        return _cubic_record(G, probe, grid, side, mode)

    def remember(record):
        if record is None:
            return
        keepers.setdefault(record["signature"], record)
        if len(keepers) > 6:
            worst = max(keepers, key=lambda s: keepers[s]["cost"])
            del keepers[worst]

    for _ in range(32):
        remember(inspect(fresh()))
    if not keepers:
        raise ValueError("no connected cubic component start")
    current = min(keepers.values(), key=lambda r: r["cost"])

    for step in range(72):
        grid = _cubic_copy(current["grid"])
        j = rng.randrange(width)
        for i in range(len(support)):
            grid[i][j] = draw(i, j)
        proposed = inspect(grid)
        if proposed is None:
            continue
        delta = proposed["cost"] - current["cost"]
        temperature = G.q * (6 * (1 - step / 72.0) ** 2 + 0.1)
        if delta <= 0 or rng.random() < math.exp(
                min(0.0, -delta / temperature)):
            current = proposed
            remember(proposed)

    def screen(record, trials, budget):
        # These are exact ranks and exact <=4 exclusions on VERIFIED rows.
        rank = gf2_rank(record["rows"])
        if rank < len(record["rows"]) - 4:
            return False
        diagnostic = _kernel_screen(
            record["columns"], len(record["rows"]))
        if diagnostic <= 4:
            return False
        record["rank"] = rank
        record["kernel_diagnostic"] = diagnostic
        _cubic_probe(rng, record, trials)
        _cubic_completion(rng, record, G.q, budget)
        return True

    survivors = []
    for record in sorted(keepers.values(), key=lambda r: r["cost"]):
        if screen(record, 8, 3500):
            survivors.append(record)
    if not survivors:
        # Fresh, independently labelled fallback attempts still receive
        # every screen; no unchecked matrix is returned.
        for _ in range(32):
            record = inspect(fresh())
            if record is not None and screen(record, 8, 3500):
                survivors.append(record)
                if len(survivors) >= 3:
                    break
    if not survivors:
        raise ValueError("no cubic component passed rank/kernel-four gates")

    survivors.sort(key=_cubic_merit, reverse=True)
    current = survivors[0]
    selected = survivors[:4]
    seen = {r["signature"] for r in survivors}
    history = set()

    for step in range(14):
        history.update(w for w in current["words"]
                       if w.bit_count() <= 14)
        history = set(sorted(history, key=int.bit_count)[:64])
        light = sorted(current["words"], key=int.bit_count)
        if light and rng.random() < 0.85:
            occupied = sorted({v // G.q for v in _bits(
                rng.choice(light[:8]))})
            j = rng.choice(occupied)
        else:
            j = rng.randrange(width)
        grid = _cubic_copy(current["grid"])
        for i in range(len(support)):
            grid[i][j] = draw(i, j)
        proposed = inspect(grid)
        if proposed is None or proposed["signature"] in seen:
            continue
        seen.add(proposed["signature"])
        if proposed["cycles"][0] > current["cycles"][0] + G.q:
            continue
        for word in history:
            if _cubic_syndrome(proposed["columns"], word) == 0:
                _cubic_note(proposed, word)
        if not screen(proposed, 6, 2500):
            continue
        if (_cubic_merit(proposed) >= _cubic_merit(current)
                or (proposed["d"] == current["d"]
                    and rng.random() < 0.12)):
            current = proposed
        selected.append(proposed)
        selected.sort(key=_cubic_merit, reverse=True)
        del selected[6:]

    finalists = []
    signatures = set()
    for record in sorted(selected + [current],
                         key=_cubic_merit, reverse=True):
        if record["signature"] in signatures:
            continue
        signatures.add(record["signature"])
        # Independent deeper challenges can only lower an upper reading.
        _cubic_completion(rng, record, G.q, 18000)
        _cubic_probe(rng, record, 48)
        for word in record["words"]:
            assert all((row & word).bit_count() % 2 == 0
                       for row in record["rows"])
        finalists.append(record)
        if len(finalists) >= 3:
            break
    finalists.sort(key=_cubic_merit, reverse=True)
    return finalists


def _component_data(rows, width):
    """Statistics of components extracted from the verified LP lift."""
    columns = [0] * width
    for r, row in enumerate(rows):
        for v in _bits(row):
            columns[v] |= 1 << r
    if not all(columns) or len(set(columns)) != width:
        return None

    adjacency = [0] * len(rows)
    check_pairs = {}
    four = 0
    forced = 0
    for syndrome in columns:
        ns = list(_bits(syndrome))
        degree = len(ns)
        forced += degree * (degree - 1) * (degree - 2) // 6
        for a, u in enumerate(ns):
            adjacency[u] |= syndrome ^ (1 << u)
            for v in ns[:a]:
                key = v * len(rows) + u
                old = check_pairs.get(key, 0)
                four += old
                check_pairs[key] = old + 1

    visited = frontier = 1
    while frontier:
        reached = 0
        for u in _bits(frontier):
            reached |= adjacency[u]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()
    # Exact six-cycle count only when four=0; otherwise a surrogate.
    return columns, four, max(0, triangles - forced)


def _kernel_screen(columns, checks):
    """Exact <=4 screen; detected connected-triple five/six witnesses.

    Returning seven means these enumerations found no word <=6,
    not an exhaustive general weight-five/six exclusion.
    """
    width = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != width:
        return 2

    pairs = {}
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs[syndrome] = (1 << a) | (1 << b)

    incidence = [[] for _ in range(checks)]
    for v, syndrome in enumerate(columns):
        for r in _bits(syndrome):
            incidence[r].append(v)
    adjacency = [set() for _ in columns]
    for variables in incidence:
        for v in variables:
            adjacency[v].update(variables)
            adjacency[v].discard(v)

    seen = {}
    found_six = False
    for b, neighbors in enumerate(adjacency):
        ns = sorted(neighbors)
        for t, a in enumerate(ns):
            for c in ns[t + 1:]:
                word = (1 << a) | (1 << b) | (1 << c)
                syndrome = columns[a] ^ columns[b] ^ columns[c]
                pair = pairs.get(syndrome)
                if pair is not None and not (pair & word):
                    return 5
                old = seen.get(syndrome)
                if old is not None:
                    if old != word and not (old & word):
                        found_six = True
                else:
                    seen[syndrome] = word
    return 6 if found_six else 7


def _circuit_reading(rng, columns, trials):
    """Classical kernel upper reading, never a distance certificate."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for v in order:
            syndrome, word = columns[v], 1 << v
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                best = min(best, word.bit_count())
        light = sorted(basis, key=int.bit_count)[:16]
        for a in light:
            for b in basis:
                if a != b:
                    best = min(best, (a ^ b).bit_count())
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Twenty fully cubic designs on the new 0.350 bed."""
    q = G.q
    assert q == 78
    card, cohort = idx % 20, idx // 20
    profile, mode = divmod(card, 4)
    sa = [[3, 3, 3]]
    columns = _CUBIC_B[profile]
    sb = [[columns[j][i] for j in range(5)] for i in range(2)]
    assert list(map(sum, sb)) == [7, 8]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache only NATIVE group operations. All matrices, including probes,
    # are still constructed by the verified scaffold helper.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    mask = (1 << q) - 1

    for restart in range(2):
        amode = (mode + cohort + restart) % 4
        bmode = (mode + profile + 1 + cohort + restart) % 4
        aa = _cubic_pool(rng, G, probe, sa, "A", amode)
        bb = _cubic_pool(rng, G, probe, sb, "B", bmode)
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            -ab[0]["cost"] - ab[1]["cost"]), reverse=True)

        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 17 * q == 1326
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= 468 and 325 * n <= 1000 * k < 375 * n):
                continue

            # Independently confirm probe extraction against ORIGINAL-GROUP
            # final matrices, not merely against the cached integer encoding.
            actual_a = [
                sum(((hx[t] >> (j * 5 * q)) & mask) << (j * q)
                    for j in range(3))
                for t in range(q)
            ]
            bmask = (1 << (5 * q)) - 1
            actual_b = [hz[alpha * q + t] & bmask
                        for alpha in range(2) for t in range(q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert max(row.bit_count() for row in hx) == 12
            assert max(row.bit_count() for row in hz) == 11
            assert css_ok(hx, hz)
            return {
                "idx": idx, "card": card,
                "cells": {"A": A, "B": B}, "k": k,
                "component_readings": (a["d"], b["d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "component_ranks": (a["rank"], b["rank"]),
                "kernel_diagnostics": (
                    a["kernel_diagnostic"], b["kernel_diagnostic"]),
            }
    raise ValueError("screened cubic pairs missed the 0.350 rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform C2 x C22 labels reliably suppress sector-1 weight-six logicals; all five measured parent codes instead had distance six and exclusively sector-1 minimum witnesses.
# REFUTED: Noncommuting order-ten binomial directions and component screening on C11:C10 (1,3)x(1,4) produce five readings >=8; the measured selections instead returned 6,6,6,6,5, with minimum witnesses entirely in sector one.
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
