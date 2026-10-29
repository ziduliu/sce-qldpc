"""RANDOM_MUTATOR=1: replace the LLM call with the leveled random mutation
operator (ported from the standalone random arm, /tmp/rand_mapk_rep.py) while
keeping the whole openevolve selection machinery (MAP-Elites k_bin archive,
islands, migration, parent sampling) untouched. The mutator gets the same
prompt (parent program + ASSIGNMENT card), applies the matching L1/L2/L3
random mutation to the EVOLVE block, and returns a SEARCH/REPLACE diff.
Registry groups + explicit grids only - that IS the random move set."""
import os, re, ast, sys
import numpy as np

sys.path.insert(0, "/u/zdliu/lp/lifted_product_search")
sys.path.insert(0, "/u/zdliu/lp/lps_graftgate")
import random_search_all_levels as rs
import distance_probe as dp

# per-worker decorrelation: openevolve runs workers as separate processes that
# each import this module; seeding all of them identically would make two
# workers mutate the same parent into the SAME child. Mix pid + time into the
# stream (base seed still sets the experiment identity).
import time as _time
rng = np.random.default_rng([int(os.environ.get("RANDOM_MUTATOR_SEED", "7")),
                             os.getpid(), int(_time.time_ns() % (2**31))])
NCALL = [0]

REGISTRY = [
    ("cyclic", lambda q: q >= 2),
    ("abelian", lambda q: q % 3 == 0 and q >= 6),
    ("dihedral", lambda q: q % 2 == 0 and q >= 4),
    ("dicyclic", lambda q: q % 4 == 0 and q >= 8),
]
VQ = dict(REGISTRY)

def _rank(H):
    return len(dp._rref_with_pivots(np.asarray(H, np.uint8) % 2)[1])

def _rand_cell(els):
    m = int(rng.integers(1, 4))
    return [list(els[i]) for i in rng.choice(len(els), m, replace=False)]

def _nwin():
    return (int(os.environ.get("LAYOUT_N_MIN", "1000")),
            int(os.environ.get("LAYOUT_N_MAX", "1500")))

def _valid_qs(gtype, shA, shB):
    N_LO, N_HI = _nwin()
    F = shA[1] * shB[1] + shA[0] * shB[0]
    vq = VQ[gtype]
    return [q for q in range(2, max(N_HI // F, 2) + 1) if vq(q) and N_LO <= F * q <= N_HI]

def _pick_q(gtype, shA, shB):
    for q in list(reversed(_valid_qs(gtype, shA, shB)))[:12]:
        ops = rs.group_ops(gtype, q, 3)
        els = ops["elements"]
        gA = [[_rand_cell(els) for _ in range(shA[1])] for _ in range(shA[0])]
        gB = [[_rand_cell(els) for _ in range(shB[1])] for _ in range(shB[0])]
        A = rs._lift(gA, ops, "left"); B = rs._lift(gB, ops, "right")
        HX, HZ = rs._lifted_product(A, B, q)
        if HX.shape[1] - _rank(HX) - _rank(HZ) > 0:
            return dict(A=gA, B=gB, q=q, gtype=gtype)
    return None

def _parse(block):
    gt = re.search(r"GTYPE = \"([a-z]+)\"", block)
    sa = re.search(r"A_SHAPE = \(([0-9]+), *([0-9]+)\)", block)
    sb = re.search(r"B_SHAPE = \(([0-9]+), *([0-9]+)\)", block)
    ca = re.search(r"CANDIDATES = (\[.*\])\s*$", block, re.M)
    if not (gt and sa and sb and ca) or gt.group(1) not in VQ:
        return None
    try:
        cand = ast.literal_eval(ca.group(1))
    except Exception:
        return None
    if not (isinstance(cand, list) and cand and isinstance(cand[0], dict) and "A" in cand[0]):
        return None
    return dict(gtype=gt.group(1),
                shA=(int(sa.group(1)), int(sa.group(2))),
                shB=(int(sb.group(1)), int(sb.group(2))),
                A=cand[0]["A"], B=cand[0]["B"])

def _q_of(geno):
    qs = _valid_qs(geno["gtype"], geno["shA"], geno["shB"])
    return qs[-1] if qs else None

def _copy_grid(G):
    return [[[list(e) for e in cell] for cell in row] for row in G]

def _mutate(geno, card):
    if card == "1":
        q = _q_of(geno)
        if q is None:
            return None
        els = rs.group_ops(geno["gtype"], q, 3)["elements"]
        A, B = _copy_grid(geno["A"]), _copy_grid(geno["B"])
        grid = A if rng.random() < 0.5 else B
        i = int(rng.integers(len(grid))); j = int(rng.integers(len(grid[0])))
        cell = grid[i][j]
        new = list(els[int(rng.integers(len(els)))])
        if cell:
            cell[int(rng.integers(len(cell)))] = new
        else:
            cell.append(new)
        return dict(gtype=geno["gtype"], shA=geno["shA"], shB=geno["shB"], A=A, B=B)
    if card == "2":
        for _ in range(20):
            shA = (int(rng.integers(1, 4)), int(rng.integers(1, 6)))
            shB = (int(rng.integers(1, 4)), int(rng.integers(1, 6)))
            fresh = _pick_q(geno["gtype"], shA, shB)
            if fresh is None:
                continue
            if fresh["q"] == _q_of(geno):
                for (src, dst) in ((geno["A"], fresh["A"]), (geno["B"], fresh["B"])):
                    for i in range(min(len(src), len(dst))):
                        for j in range(min(len(src[0]), len(dst[0]))):
                            dst[i][j] = [list(e) for e in src[i][j]]
            return dict(gtype=geno["gtype"], shA=shA, shB=shB, A=fresh["A"], B=fresh["B"])
        return None
    if card == "G":
        # true global: shapes, group and fill all drawn fresh
        for _ in range(20):
            shA = (int(rng.integers(1, 4)), int(rng.integers(1, 6)))
            shB = (int(rng.integers(1, 4)), int(rng.integers(1, 6)))
            gname = REGISTRY[int(rng.integers(len(REGISTRY)))][0]
            fresh = _pick_q(gname, shA, shB)
            if fresh is not None:
                return dict(gtype=gname, shA=shA, shB=shB,
                            A=fresh["A"], B=fresh["B"])
        return None
    gname = REGISTRY[int(rng.integers(len(REGISTRY)))][0]
    fresh = _pick_q(gname, geno["shA"], geno["shB"])
    if fresh is None:
        return None
    return dict(gtype=gname, shA=geno["shA"], shB=geno["shB"], A=fresh["A"], B=fresh["B"])

def _render(g, card):
    cand = repr([{"A": g["A"], "B": g["B"]}])
    return ("# EVOLVE-BLOCK-START\n"
            "CONCEPT = (\"random L%s mutation on explicit grids (registry group %s)\")\n"
            "HOW_AB  = (\"uniform random cells; no design\")\n"
            "FAMILY_NAME = \"rnd_%s\"\n"
            "GTYPE = \"%s\"; A0 = 3\n"
            "CUSTOM_GROUP = None\n"
            "A_SHAPE = (%d, %d); B_SHAPE = (%d, %d)\n\n"
            "CANDIDATES = %s\n\n"
            "# EVOLVE-BLOCK-END") % (
        card, g["gtype"], g["gtype"], g["gtype"],
        g["shA"][0], g["shA"][1], g["shB"][0], g["shB"][1], cand)

def respond(prompt_text):
    NCALL[0] += 1
    # the card sits in the trailing ASSIGNMENT section; program code embedded
    # earlier in the prompt also contains LEVEL_1... strings (CONSTRUCTION_SPEC),
    # so match the ASSIGNMENT heading specifically, else take the LAST occurrence.
    m = re.search(r"# ASSIGNMENT[^\n]*\n.*?LEVEL_([123])", prompt_text, re.S)
    if not m:
        all_lv = re.findall(r"LEVEL_([123])", prompt_text)
        card = all_lv[-1] if all_lv else str(int(rng.integers(1, 4)))
    else:
        card = m.group(1)
    _force = os.environ.get("RANDOM_MUTATOR_FORCE_LEVEL", "").strip()
    if _force in ("1", "2", "3", "G"):
        card = _force
    cur = re.search(r"# Current Program\s*\n```[a-zA-Z]*\n(.*?)\n```", prompt_text, re.S)
    prog = cur.group(1) if cur else prompt_text
    bm = re.search(r"(# EVOLVE-BLOCK-START.*?# EVOLVE-BLOCK-END)", prog, re.S)
    if not bm:
        return "RANDOM_MUTATOR: no EVOLVE block found"
    old_block = bm.group(1)
    geno = _parse(old_block)
    child = _mutate(geno, card) if geno is not None else None
    if child is None:
        gname = REGISTRY[int(rng.integers(len(REGISTRY)))][0]
        fresh = _pick_q(gname, (2, 3), (2, 3))
        if fresh is None:
            return "RANDOM_MUTATOR: no viable mutation"
        child = dict(gtype=gname, shA=(2, 3), shB=(2, 3), A=fresh["A"], B=fresh["B"])
    new_block = _render(child, card)
    with open("/tmp/random_mutator_calls.log", "a") as f:
        f.write("call=%d card=L%s parsed=%s child=%s\n" %
                (NCALL[0], card, geno is not None, child["gtype"]))
    return ("<<<<<<< SEARCH\n" + old_block + "\n=======\n" +
            new_block + "\n>>>>>>> REPLACE\n")
