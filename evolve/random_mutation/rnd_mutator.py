"""Random-mutation arm: the proposal step of the evolutionary loop with the
LLM replaced by a random draw at the dealt card's level. The engine is
unchanged; only the child is drawn at random, from the layers the card allows.

  L1 (element model)  group and shapes kept; the filling is re-rolled (a new
                      BASE_SEED; the scaffold's per-slot rng is
                      random.Random(BASE_SEED * 10007 + slot)).
  L2 (shapes)         a new (A_SHAPE, B_SHAPE) drawn uniformly from the shape
                      families (sides 1..8, weight-feasible, A/B roles
                      re-randomised) whose rank-count rate
                      (ca-ra)(cb-rb)/(ca*cb+ra*rb) lies in the named cell's
                      band; the group is kept if its order still puts n in the
                      window, otherwise the order is re-drawn keeping the
                      family (same quotient size r, or dicyclic).
  L3 (group)          a new group from the library at an order chosen by the
                      order rule (RND_ORDER_RULE); shapes kept.

Group library without RND_LIBRARY: every split metacyclic presentation
Z_m x|_s Z_r (m*r = q, gcd(s, m) = 1, s^r = 1 mod m; r = 1 cyclic, s = 1 direct
product) plus the dicyclic groups Dic_n (q = 4n). Draws are uniform over
presentations of the chosen order.

The child is emitted as the SEARCH/REPLACE diff the engine expects. All draws
come from one random.Random handed in by the caller (the server keeps one
generator for its whole life; per-request generators seeded from the clock
would repeat under concurrent calls).
"""
import math
import re

N_MIN, N_MAX = 100, 600          # size window (CM_NMIN / CM_NMAX of the lines)
W_MAX = 12                       # row-weight cap (CM_WMAX)
SIDE_MAX = 8                     # protograph sides 1..8
CELL_MAX = 5                     # a cell holds up to 5 DISTINCT elements ...
Q_MIN = CELL_MAX                 # ... so the scaffold's CELL loop needs q >= 5
Q_TOP = 3                        # orders are drawn from the Q_TOP largest admissible ones

# Order rule. RND_ORDER_RULE=top (default: order uniform over the Q_TOP largest admissible) or
# llm_n: a target n is drawn from the block lengths of the children of the gpt-6.1-sol run
# (RND_N_DIST json, key RND_N_KEY) and the admissible order with n = f*q nearest to it is taken;
# llm_n also excludes the parent's group at L3.
RND_ORDER_RULE = __import__("os").environ.get("RND_ORDER_RULE", "top")
RND_N_DIST = __import__("os").environ.get("RND_N_DIST", "/u/zdliu/lp/circmean/llm_n_distribution.json")
RND_N_KEY = __import__("os").environ.get("RND_N_KEY", "fom")
_N_SAMPLES = None


# v5 rules, switched on by RND_V5=1: draw_child_v5 (end of file) draws the child instead of
# draw_child, following the rules the prompt states:
#   shapes   sides 1..8, rank-count bound >= 0 instead of > 0, so (1,1)x(1,1) and the other
#            bound-0 shapes are drawable; they belong to the open-ended lowest cell, whose band is
#            (-inf, edge). Negative bounds stay excluded.
#            The child's CONCEPT, GROUP and SHAPES lines are rewritten to say that its group and
#            shapes were drawn by the random arm.
#   level 2  the parent's group is kept whenever its order is admissible on the new shapes;
#            otherwise the order moves to the admissible order of the same family nearest to a
#            block length drawn from the gpt-6.1-sol children. Shapes are drawn among the band's
#            shapes on which the family has an admissible order, the parent's own shape excluded;
#            if there is none, only the filling is re-rolled.
#            Family: a template parent keeps its template (metacyclic with any m, r, s, or
#            dicyclic); a library parent keeps the family of its GAP StructureDescription (numbers
#            replaced by '#'): the metacyclic forms C#, C# x C#, C# : C#, D#, Q#, QD# form one
#            family, every other form is its own family (RND_SD_PATH).
#   level 3  a group other than the parent's: the eligible choices are listed before the draw. A
#            library parent excludes its own group; a template parent excludes the library group it
#            is isomorphic to (GAP IdGroup of every template presentation, RND_TMPL_PATH). The shapes
#            are also redrawn, uniformly over all shapes, with probability RND_L3_SHAPE_P (the
#            share of level-3 children of the gpt-6.1-sol run of the same objective that changed
#            their shapes), and always when the parent's shapes admit no other group.
RND_V5 = __import__("os").environ.get("RND_V5", "") == "1"
RND_SD_PATH = __import__("os").environ.get("RND_SD_PATH", "/u/zdliu/lp/circmean/smallgrp_sd.tsv")
RND_TMPL_PATH = __import__("os").environ.get("RND_TMPL_PATH", "/u/zdliu/lp/circmean/template_ids.tsv")
RND_L3_SHAPE_P = float(__import__("os").environ.get("RND_L3_SHAPE_P", "0")) if RND_V5 else 0.0
OPEN_LO = -1e9                   # lower end of the open-ended lowest cell under v5
METACYCLIC_FORMS = frozenset(["C#", "C# x C#", "C# : C#", "D#", "Q#", "QD#"])


def llm_n_samples():
    global _N_SAMPLES
    if _N_SAMPLES is None:
        import json
        _N_SAMPLES = sorted(int(x) for x in json.load(open(RND_N_DIST))[RND_N_KEY])
        assert len(_N_SAMPLES) >= 100
    return _N_SAMPLES


def llm_n_cdf(n):
    """Fraction of the LLM run's children with block length <= n."""
    xs = llm_n_samples()
    import bisect
    return bisect.bisect_right(xs, n) / len(xs)


def draw_order(ra, ca, rb, cb, rng, admits=lambda q: True):
    """llm_n rule: the admissible order (passing admits) whose n = f*q is nearest to a block
    length drawn from the LLM run's children; ties go to the larger order. None if no order."""
    f = shape_factor(ra, ca, rb, cb)
    orders = [q for q in admissible_orders(ra, ca, rb, cb) if admits(q)]
    if not orders:
        return None
    target = rng.choice(llm_n_samples())
    return min(orders, key=lambda q: (abs(f * q - target), -q))


_RE_SEED = re.compile(r"^BASE_SEED = (\d+)\s*$", re.M)
_RE_ASHAPE = re.compile(r"^A_SHAPE = \((\d+), (\d+)\)(.*)$", re.M)
_RE_BSHAPE = re.compile(r"^B_SHAPE = \((\d+), (\d+)\)(.*)$", re.M)
_RE_FAMILY = re.compile(r'^FAMILY_NAME = "([^"\n]*)"\s*$', re.M)
_RE_GROUP = re.compile(r'^CUSTOM_GROUP = """\n(.*?)\n"""$', re.M | re.S)
_RE_MRS = re.compile(r"^\s*m, r, s = (\d+), (\d+), (\d+)\s*$", re.M)
_RE_DIC = re.compile(r"^\s*n2 = (\d+)\s*$", re.M)
_RE_SG = re.compile(r"^\s*sg = (\d+), (\d+)\s*$", re.M)
_RE_VALIDQ = re.compile(r"^\s*return q == (\d+)\s*$", re.M)
# the scaffold's weight-cap line: "W_MAX = int(os.environ.get("CM_WMAX", "12"))" in a
# seed, "W_MAX = min(int(os.environ.get("CM_WMAX", "12")), 11)  # ..." after a mutation
_RE_WMAX = re.compile(r"^W_MAX = .*$", re.M)
# Every child draws its own row-weight cap uniformly from W_CAPS (RND_WCAPS, comma
# separated; RND_WCAPS=12 fixes every child at the run's cap). Caps below the shape's
# lightest row (ca+rb, cb+ra) are skipped; roll_supports repairs the filling down to the cap.
W_CAPS = tuple(int(x) for x in __import__("os").environ.get("RND_WCAPS", "8,9,10,11,12").split(","))
# RND_LIBRARY=smallgrp replaces the two presentation templates below by the GAP small groups
# library, exported as permutation generators (gap_library/export.g: every group of order
# 5..299 except 256, plus a fixed uniform sample of 300 groups of order 256). A level-3 draw
# is then uniform over all library groups of the drawn order.
RND_LIBRARY = __import__("os").environ.get("RND_LIBRARY", "")
RND_LIB_PATH = __import__("os").environ.get("RND_LIB_PATH", "/u/zdliu/lp/circmean/smallgrp_lib.jsonl")
if RND_V5 and (RND_LIBRARY != "smallgrp" or RND_ORDER_RULE != "llm_n"):
    # the v5 level-3 exclusion compares library presentations; it is defined only with the library
    raise RuntimeError("RND_V5=1 requires RND_LIBRARY=smallgrp and RND_ORDER_RULE=llm_n")
_SG = None


def _sg_lib():
    global _SG
    if _SG is None:
        import json as _json
        _SG = {}
        for line in open(RND_LIB_PATH):
            if line.strip():
                r = _json.loads(line)
                _SG.setdefault(int(r["q"]), {})[int(r["id"])] = r
    return _SG


def smallgrp_presentations(q):
    return [("sg", q, i) for i in sorted(_sg_lib().get(q, {}))]


# --------------------------------------------------------------------------
# group library
# --------------------------------------------------------------------------
def metacyclic_presentations(q):
    """All (m, r, s) with m*r = q, m >= 2, gcd(s, m) = 1, s^r = 1 mod m.
    The cyclic group appears once, as (q, 1, 1)."""
    out = []
    for r in range(1, q + 1):
        if q % r:
            continue
        m = q // r
        if m < 2:
            continue
        for s in range(1, m):
            if math.gcd(s, m) == 1 and pow(s, r, m) == 1:
                out.append(("mc", m, r, s))
    return out


def dicyclic_presentations(q):
    """Dic_n for q = 4n, n >= 2 (Dic_2 is the quaternion group)."""
    if q % 4 == 0 and q >= 8:
        return [("dic", q // 4)]
    return []


def presentations(q):
    if RND_LIBRARY == "smallgrp":
        return smallgrp_presentations(q)
    return metacyclic_presentations(q) + dicyclic_presentations(q)


def group_block(pres, q):
    """The CUSTOM_GROUP string body (between the triple quotes) for a
    presentation - the seeds' metacyclic template with the numbers swapped,
    or the dicyclic template. Both return elements as (a, b) tuples."""
    if pres[0] == "mc":
        _, m, r, s = pres
        assert m * r == q
        return (
            "def valid_q(q):\n"
            "    return q == %d\n"
            "\n"
            "def custom_ops(q):\n"
            "    # metacyclic Z_%d x| Z_%d: y x y^-1 = x^%d (s^r = 1 mod m);\n"
            "    # drawn by the random-mutation arm (rnd_mutator.py), not designed\n"
            "    m, r, s = %d, %d, %d\n"
            "    els = [(a, b) for b in range(r) for a in range(m)]\n"
            "\n"
            "    def mul(g, h):\n"
            "        return ((g[0] + pow(s, g[1], m) * h[0]) %% m, (g[1] + h[1]) %% r)\n"
            "\n"
            "    def inv(g):\n"
            "        bi = (-g[1]) %% r\n"
            "        return ((-pow(s, bi, m) * g[0]) %% m, bi)\n"
            "\n"
            "    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)"
            % (q, m, r, s, m, r, s))
    if pres[0] == "dic":
        n = pres[1]
        assert 4 * n == q
        return (
            "def valid_q(q):\n"
            "    return q == %d\n"
            "\n"
            "def custom_ops(q):\n"
            "    # dicyclic Dic_%d (order 4n): x^(2n) = 1, y^2 = x^n, y x y^-1 = x^-1;\n"
            "    # element (a, b) = x^a y^b, a mod 2n, b in {0, 1};\n"
            "    # drawn by the random-mutation arm (rnd_mutator.py), not designed\n"
            "    n2 = %d\n"
            "    els = [(a, b) for b in range(2) for a in range(n2)]\n"
            "\n"
            "    def mul(g, h):\n"
            "        a, b = g\n"
            "        c, d = h\n"
            "        return ((a + (c if b == 0 else -c) + (n2 // 2 if b + d == 2 else 0)) %% n2,\n"
            "                (b + d) %% 2)\n"
            "\n"
            "    def inv(g):\n"
            "        a, b = g\n"
            "        return ((-a) %% n2, 0) if b == 0 else ((a + n2 // 2) %% n2, 1)\n"
            "\n"
            "    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)"
            % (q, n, 2 * n))
    if pres[0] == "sg":
        _, qq, i = pres
        assert qq == q
        r = _sg_lib()[q][i]
        gens = ", ".join("(" + ", ".join(str(x) for x in g) + ")" for g in r["gens"])
        return (
            "def valid_q(q):\n"
            "    return q == %d\n"
            "\n"
            "def custom_ops(q):\n"
            "    # SmallGroup(%d, %d) of the GAP small groups library, as a permutation group on %d\n"
            "    # points (generators below, 0-based, composed as (g*h)(x) = g[h[x]]);\n"
            "    # drawn by the random-mutation arm (rnd_mutator.py, full-library variant), not designed\n"
            "    sg = %d, %d\n"
            "    deg = %d\n"
            "    gens = [%s]\n"
            "    ident = tuple(range(deg))\n"
            "\n"
            "    def mul(g, h):\n"
            "        return tuple(g[x] for x in h)\n"
            "\n"
            "    def inv(g):\n"
            "        r = [0] * deg\n"
            "        for i, gi in enumerate(g):\n"
            "            r[gi] = i\n"
            "        return tuple(r)\n"
            "\n"
            "    els = [ident]\n"
            "    seen = {ident}\n"
            "    frontier = [ident]\n"
            "    while frontier:\n"
            "        nxt = []\n"
            "        for x in frontier:\n"
            "            for s in gens:\n"
            "                y = mul(x, s)\n"
            "                if y not in seen:\n"
            "                    seen.add(y)\n"
            "                    els.append(y)\n"
            "                    nxt.append(y)\n"
            "        frontier = nxt\n"
            "    if len(els) != q:\n"
            "        raise ValueError(\"group closure has %%d elements, expected %%d\" %% (len(els), q))\n"
            "    return dict(elements=els, ident=ident, mul=mul, inv=inv)"
            % (q, q, i, r["deg"], q, i, r["deg"], gens))
    raise ValueError(pres)


def group_tag(pres):
    if pres[0] == "mc":
        return "m%dr%ds%d" % pres[1:]
    if pres[0] == "sg":
        return "sg%d_%d" % pres[1:]
    return "dic%d" % pres[1]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def shape_factor(ra, ca, rb, cb):
    return ca * cb + ra * rb


def shape_rate(ra, ca, rb, cb):
    return (ca - ra) * (cb - rb) / shape_factor(ra, ca, rb, cb)


def weight_feasible(ra, ca, rb, cb):
    """Every cell carries >= 1 term, so the lightest HX row is ca + rb and the
    lightest HZ row is cb + ra."""
    return ca + rb <= W_MAX and cb + ra <= W_MAX


def admissible_orders(ra, ca, rb, cb):
    """Orders q with n = f*q in [N_MIN, N_MAX) - the scaffold's _choose_q
    range - and q >= Q_MIN: the scaffold's CELL draws up to CELL_MAX distinct
    elements per cell and loops forever when the group has fewer."""
    f = shape_factor(ra, ca, rb, cb)
    return [q for q in range(max(Q_MIN, N_MIN // f), N_MAX // f + 1)
            if N_MIN <= f * q < N_MAX]


def top_orders(ra, ca, rb, cb, admits=lambda q: True, k=Q_TOP):
    """The k largest admissible orders that admit a presentation (admits(q)),
    largest first. The scaffold's own _choose_q also takes the largest valid
    order; drawing over the whole window would spend most draws on small lifts.
    Uniform over these k orders, then uniform over presentations."""
    out = []
    for q in sorted(admissible_orders(ra, ca, rb, cb), reverse=True):
        if admits(q):
            out.append(q)
        if len(out) >= k:
            break
    return out


def shape_families():
    """The shape prior: sides 1..SIDE_MAX, rank-count bound
    > 0, some admissible order, all-ones grid inside the weight cap; A/B swap
    and double transpose counted once (the roles are re-randomised on draw)."""
    seen, fams = set(), []
    for ra in range(1, SIDE_MAX + 1):
        for ca in range(1, SIDE_MAX + 1):
            for rb in range(1, SIDE_MAX + 1):
                for cb in range(1, SIDE_MAX + 1):
                    if (ca - ra) * (cb - rb) <= 0:
                        continue
                    if not weight_feasible(ra, ca, rb, cb):
                        continue
                    if not admissible_orders(ra, ca, rb, cb):
                        continue
                    a = tuple(sorted([(ra, ca), (rb, cb)]))
                    b = tuple(sorted([(ca, ra), (cb, rb)]))
                    key = min(a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    fams.append(key)
    return fams


_FAMS = None


def shapes_in_band(lo, hi):
    """Families whose rank-count rate lies in [lo, hi) (the rate is invariant
    under A/B swap and double transpose, so it is a family property)."""
    global _FAMS
    if _FAMS is None:
        _FAMS = shape_families()
    return [fam for fam in _FAMS if lo <= shape_rate(fam[0][0], fam[0][1], fam[1][0], fam[1][1]) < hi]


# --------------------------------------------------------------------------
# parsing the parent
# --------------------------------------------------------------------------
class ParseError(ValueError):
    pass


def parse_program(code):
    """The bed of a seed-scaffold program: seed line, shapes, family name and
    the CUSTOM_GROUP block (raw text plus, when it is one of our two templates,
    the presentation)."""
    m = _RE_SEED.search(code)
    if not m:
        raise ParseError("no BASE_SEED line")
    p = {"seed": int(m.group(1)), "seed_line": m.group(0)}
    for key, rx in (("A", _RE_ASHAPE), ("B", _RE_BSHAPE)):
        m = rx.search(code)
        if not m:
            raise ParseError("no %s_SHAPE line" % key)
        p[key] = (int(m.group(1)), int(m.group(2)))
        p[key + "_line"] = m.group(0)
        p[key + "_tail"] = m.group(3)
    m = _RE_FAMILY.search(code)
    p["family"] = m.group(1) if m else None
    p["family_line"] = m.group(0) if m else None
    m = _RE_WMAX.search(code)
    p["wmax_line"] = m.group(0) if m else None
    m = _RE_GROUP.search(code)
    if not m:
        raise ParseError("no CUSTOM_GROUP block")
    p["group_text"] = m.group(0)
    body = m.group(1)
    mq = _RE_VALIDQ.search(body)
    p["q"] = int(mq.group(1)) if mq else None
    mm = _RE_MRS.search(body)
    md = _RE_DIC.search(body)
    ms = _RE_SG.search(body)
    if ms:
        p["pres"] = ("sg", int(ms.group(1)), int(ms.group(2)))
        if p["q"] is None:
            p["q"] = int(ms.group(1))
    elif mm:
        p["pres"] = ("mc", int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
        if p["q"] is None:
            p["q"] = p["pres"][1] * p["pres"][2]
    elif md:
        p["pres"] = ("dic", int(md.group(1)) // 2)
        if p["q"] is None:
            p["q"] = int(md.group(1)) * 2
    else:
        p["pres"] = None
    return p


# --------------------------------------------------------------------------
# the draw
# --------------------------------------------------------------------------
def _same_family(pres, q):
    """Presentations of order q in the parent's family: same quotient size r
    for a metacyclic parent (the (1, q, 1) cyclic seeds count as r = q -> the
    cyclic group of order q), dicyclic for a dicyclic parent."""
    if pres is None or pres[0] == "sg" or RND_LIBRARY == "smallgrp":
        return presentations(q)
    if pres[0] == "dic":
        return dicyclic_presentations(q)
    _, m, r, s = pres
    if m == 1:                       # seed form (1, q, 1) = cyclic
        return [("mc", q, 1, 1)]
    return [x for x in metacyclic_presentations(q) if x[2] == r]


def draw_child(p, level, band, rng):
    """Decide the child's bed. Returns (new_seed, new_shapes, new_pres, new_q,
    note). new_shapes / new_pres are None when that layer is kept."""
    if RND_V5:
        return draw_child_v5(p, level, band, rng)
    new_seed = rng.randrange(1, 2 ** 31)
    ra, ca = p["A"]
    rb, cb = p["B"]
    if level == 2 and band is not None:
        lo, hi = band
        # measured k exceeds the rank-count bound, so a shape whose bound sits one cell
        # below the target usually lands in the target; widen the band downward by one cell
        lo = max(1e-9, lo - 0.05)
        cands = shapes_in_band(lo, hi)
        if not cands:
            return new_seed, None, None, None, "L2: no feasible shape in [%g, %g) - re-rolled only" % (lo, hi)
        (ra, ca), (rb, cb) = rng.choice(cands)
        if rng.random() < 0.5:                       # A/B roles at random
            (ra, ca), (rb, cb) = (rb, cb), (ra, ca)
        orders = admissible_orders(ra, ca, rb, cb)
        # top rule: keep the parent's group only when its order is among the new shape's
        # top-Q_TOP admissible orders (otherwise many children stay at small n)
        if RND_ORDER_RULE == "llm_n":
            # keep the parent's group with the probability that an LLM child is at most as long
            # as n = f*q would be
            f = shape_factor(ra, ca, rb, cb)
            if p["pres"] is not None and p["q"] in orders and rng.random() < llm_n_cdf(f * p["q"]):
                return new_seed, (ra, ca, rb, cb), None, p["q"], "L2: shape -> (%d,%d)x(%d,%d), group kept (q=%d, n=%d, llm_n)" % (ra, ca, rb, cb, p["q"], f * p["q"])
            q = draw_order(ra, ca, rb, cb, rng, admits=lambda q: bool(_same_family(p["pres"], q)))
            if q is not None:
                pres = rng.choice(_same_family(p["pres"], q))
                how = "same family, order matched to the LLM block lengths"
            else:
                q = draw_order(ra, ca, rb, cb, rng)
                pres = rng.choice(presentations(q))
                how = "no same-family order admissible, drawn from the whole library, order matched to the LLM block lengths"
            return new_seed, (ra, ca, rb, cb), pres, q, "L2: shape -> (%d,%d)x(%d,%d), %s: %s q=%d" % (ra, ca, rb, cb, how, group_tag(pres), q)
        if p["pres"] is not None and p["q"] in orders and p["q"] in top_orders(ra, ca, rb, cb):
            return new_seed, (ra, ca, rb, cb), None, p["q"], "L2: shape -> (%d,%d)x(%d,%d), group kept (q=%d fits)" % (ra, ca, rb, cb, p["q"])
        tops = top_orders(ra, ca, rb, cb, admits=lambda q: bool(_same_family(p["pres"], q)))
        if tops:
            q = rng.choice(tops)
            pres = rng.choice(_same_family(p["pres"], q))
            how = "same family, order re-drawn from the top %d" % len(tops)
        else:
            tops = top_orders(ra, ca, rb, cb)
            q = rng.choice(tops)
            pres = rng.choice(presentations(q))
            how = "no same-family order admissible, drawn from the whole library (top %d orders)" % len(tops)
        return new_seed, (ra, ca, rb, cb), pres, q, "L2: shape -> (%d,%d)x(%d,%d), %s: %s q=%d" % (ra, ca, rb, cb, how, group_tag(pres), q)
    if level == 3:
        if RND_ORDER_RULE == "llm_n":
            # a parent on a template presentation ("mc"/"dic", the
            # seeds and their level-1/2 descendants) can never equal a library draw ("sg", ...),
            # so for such parents the parent's ORDER is excluded instead; for library parents the
            # exact (order, presentation) is excluded by redrawing.
            par_sg = bool(p["pres"]) and p["pres"][0] == "sg"
            q = pres = None
            how = "parent's group excluded"
            if not par_sg:
                q = draw_order(ra, ca, rb, cb, rng, admits=lambda qq: qq != p["q"])
                how = "parent's order %d excluded (template parent)" % p["q"]
                if q is None:                 # the parent's order is the only admissible one
                    q = draw_order(ra, ca, rb, cb, rng)
                    how = "parent's order kept (only admissible order)"
                pres = rng.choice(presentations(q))
                attempt = 0
            else:
                for attempt in range(64):
                    q = draw_order(ra, ca, rb, cb, rng)
                    pres = rng.choice(presentations(q))
                    if q != p["q"] or pres != p["pres"]:
                        break
                else:
                    how = "parent's group NOT excluded after 64 draws"
            return new_seed, None, pres, q, "L3: group -> %s q=%d (order matched to the LLM block lengths, presentation uniform over %d, %s, %d draw%s)" % (group_tag(pres), q, len(presentations(q)), how, attempt + 1, "" if attempt == 0 else "s")
        tops = top_orders(ra, ca, rb, cb)
        q = rng.choice(tops)
        pres = rng.choice(presentations(q))
        return new_seed, None, pres, q, "L3: group -> %s q=%d (order uniform over the top %d admissible %s, presentation uniform over %d)" % (group_tag(pres), q, len(tops), tops, len(presentations(q)))
    return new_seed, None, None, None, "L1: filling re-rolled (BASE_SEED %d -> %d)" % (p["seed"], new_seed)


def _diff_block(old, new):
    return "<<<<<<< SEARCH\n%s\n=======\n%s\n>>>>>>> REPLACE\n" % (old, new)


def make_response(code, level, band, rng):
    """The full 'LLM response': header lines the engine logs / parses, then
    the SEARCH/REPLACE blocks. Raises ParseError when the parent cannot be
    read (the server then answers with no diff and the round fails cleanly)."""
    p = parse_program(code)
    new_seed, shapes, pres, q, note = draw_child(p, level, band, rng)
    blocks = [_diff_block(p["seed_line"], "BASE_SEED = %d" % new_seed)]
    changes = ["filling"]
    if shapes is not None:
        ra, ca, rb, cb = shapes
        blocks.append(_diff_block(p["A_line"], "A_SHAPE = (%d, %d)%s" % (ra, ca, p["A_tail"])))
        blocks.append(_diff_block(p["B_line"], "B_SHAPE = (%d, %d)%s" % (rb, cb, p["B_tail"])))
        changes.append("shapes")
    else:
        ra, ca = p["A"]
        rb, cb = p["B"]
    if pres is not None:
        blocks.append(_diff_block(p["group_text"], 'CUSTOM_GROUP = """\n%s\n"""' % group_block(pres, q)))
        changes.append("group")
        tag = group_tag(pres)
    else:
        tag = group_tag(p["pres"]) if p["pres"] else "grp"
    # the child's row-weight cap: uniform over the caps the shape can meet
    # (its lightest rows weigh ca+rb and cb+ra); the scaffold's roll_supports
    # then repairs every filling down to this cap
    lightest = max(ca + rb, cb + ra)
    caps = [w for w in W_CAPS if w >= lightest] or [W_MAX]
    wcap = rng.choice(caps)
    if p.get("wmax_line"):
        blocks.append(_diff_block(
            p["wmax_line"],
            'W_MAX = min(int(os.environ.get("CM_WMAX", "%d")), %d)  # row-weight cap drawn from %s (random arm)'
            % (W_MAX, wcap, list(W_CAPS))))
        changes.append("weight cap %d" % wcap)
    fam = ("rnd_%s_%d-%dx%d-%d_w%d" if RND_V5 else "rnd_%s_%d%dx%d%d_w%d") % (tag, ra, ca, rb, cb, wcap)
    if RND_V5:
        qq = q if q is not None else p["q"]
        for key, text in (("CONCEPT", "Random-mutation arm, not designed: %s of order %s, shapes (%d,%d)x(%d,%d), "
                                      "row-weight cap %d; this child was drawn at card level %d." % (tag, qq, ra, ca, rb, cb, wcap, level)),
                          ("GROUP", "Drawn at random by the random-mutation arm (rnd_mutator.py v5); no reason."),
                          ("SHAPES", "Drawn at random by the random-mutation arm (rnd_mutator.py v5); no reason.")):
            m = re.search(r'^%s = \(".*"\)$' % key, code, re.M)
            if m:
                blocks.append(_diff_block(m.group(0), '%s = ("%s")' % (key, text)))
    if p["family_line"]:
        blocks.append(_diff_block(p["family_line"], 'FAMILY_NAME = "%s"' % fam))
    note = "%s; row-weight cap %d (drawn from %s, lightest rows %d)" % (note, wcap, list(caps), lightest)
    head = ["CHANGES: %s" % ", ".join(changes)]
    if band is not None:
        head.append("TARGET CELL: [%.4f, %.4f)" % ((max(band[0], 0.0), band[1]) if RND_V5 else band))
    head.append("REASON: random-mutation arm - %s" % note)
    head.append("CONCEPT: random draw at card level L%d; %s, shapes (%d,%d)x(%d,%d), q=%s" % (
        level, tag, ra, ca, rb, cb, q if q is not None else p["q"]))
    head.append("WHY: no reason - this arm is the random control")
    head.append("DIFFERS FROM PARENT: %s" % note)
    return "\n".join(head) + "\n\n" + "\n".join(blocks), dict(
        level=level, band=band, note=note, seed=new_seed, shapes=shapes,
        pres=pres, q=q, family=fam, parent_family=p["family"], wcap=wcap)


# --------------------------------------------------------------------------
# v5 (RND_V5=1)
# --------------------------------------------------------------------------
_SDPAT = None


def _sd_patterns():
    """(q, id) -> family of the library group, and family -> {q: [ids]}. The structure description
    file (q <TAB> id <TAB> StructureDescription) must list exactly the groups of the library."""
    global _SDPAT
    if _SDPAT is None:
        by_group, by_fam = {}, {}
        for ln, line in enumerate(open(RND_SD_PATH), 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit() or not parts[2].strip():
                raise RuntimeError("RND_SD_PATH line %d is malformed: %r" % (ln, line[:80]))
            q, i = int(parts[0]), int(parts[1])
            if (q, i) in by_group:
                raise RuntimeError("RND_SD_PATH line %d repeats group (%d, %d)" % (ln, q, i))
            form = re.sub(r"\d+", "#", parts[2].strip())
            fam = "metacyclic" if form in METACYCLIC_FORMS else "form:" + form
            by_group[(q, i)] = fam
            by_fam.setdefault(fam, {}).setdefault(q, []).append(i)
        lib = _sg_lib()
        keys = {(q, i) for q in lib for i in lib[q]}
        if keys != set(by_group):
            raise RuntimeError("RND_SD_PATH does not match the library: %d missing, %d extra"
                               % (len(keys - set(by_group)), len(set(by_group) - keys)))
        for fam in by_fam:
            for q in by_fam[fam]:
                by_fam[fam][q].sort()
        _SDPAT = (by_group, by_fam)
    return _SDPAT


_TMPL = None


def template_ids():
    """(template presentation) -> id of the isomorphic group in the GAP small groups library
    (RND_TMPL_PATH: tag q p1 p2 p3 id). Must cover every template presentation of orders 5..299."""
    global _TMPL
    if _TMPL is None:
        t = {}
        for ln, line in enumerate(open(RND_TMPL_PATH), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6 or parts[0] not in ("mc", "dic"):
                raise RuntimeError("RND_TMPL_PATH line %d is malformed: %r" % (ln, line[:80]))
            tag, q, a1, a2, a3, gid = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            key = ("mc", a1, a2, a3) if tag == "mc" else ("dic", a1)
            if key in t:
                raise RuntimeError("RND_TMPL_PATH line %d repeats %s" % (ln, key))
            if q != (a1 * a2 if tag == "mc" else 4 * a1) or gid < 1:
                raise RuntimeError("RND_TMPL_PATH line %d: order or id inconsistent: %r" % (ln, line[:80]))
            lib = _sg_lib()
            # every library order except 256 holds all its groups, so the twin must be there; at order
            # 256 the library is a sample of 300 groups and the twin may be absent (nothing to exclude)
            if q != 256 and (q not in lib or gid not in lib[q]):
                raise RuntimeError("RND_TMPL_PATH line %d: SmallGroup(%d, %d) is not in the library" % (ln, q, gid))
            t[key] = (q, gid)
        need = [x for q in range(Q_MIN, N_MAX // 2) for x in metacyclic_presentations(q) + dicyclic_presentations(q)]
        missing = [x for x in need if x not in t]
        if missing:
            raise RuntimeError("RND_TMPL_PATH lacks %d template presentations, e.g. %s" % (len(missing), missing[:3]))
        _TMPL = t
    return _TMPL


def library_twin(pres):
    """The library presentation isomorphic to the parent's group: the parent itself for a library
    group, ("sg", q, IdGroup) for a template."""
    if pres is None or pres[0] == "sg":
        return pres
    if pres[0] == "mc" and pres[1] == 1:     # the seeds' cyclic form (1, q, 1) is (q, 1, 1)
        pres = ("mc", pres[2], 1, 1)
    q, gid = template_ids()[pres]
    return ("sg", q, gid)


def family_key(pres):
    if pres is None:
        return None
    if pres[0] in ("mc", "dic"):
        return pres[0]
    return _sd_patterns()[0][(pres[1], pres[2])]


def family_members(pres, q):
    """Presentations of order q in the parent's family (v5)."""
    if pres is None:
        return presentations(q)
    if pres[0] == "mc":
        return metacyclic_presentations(q)
    if pres[0] == "dic":
        return dicyclic_presentations(q)
    return [("sg", q, i) for i in _sd_patterns()[1][family_key(pres)].get(q, [])]


_FAMQ = {}


def family_has_order(pres, q):
    k = (family_key(pres), q)
    if k not in _FAMQ:
        _FAMQ[k] = bool(family_members(pres, q))
    return _FAMQ[k]


def shape_key(ra, ca, rb, cb):
    """The family key of shape_families (A/B swap and double transpose identified)."""
    return min(tuple(sorted([(ra, ca), (rb, cb)])), tuple(sorted([(ca, ra), (cb, rb)])))


def _flat(fm):
    return fm[0][0], fm[0][1], fm[1][0], fm[1][1]


_FAMS5 = None


def shape_families_v5():
    """v5 shape prior: as shape_families() with rank-count bound >= 0 instead of > 0."""
    global _FAMS5
    if _FAMS5 is None:
        seen, fams = set(), []
        for ra in range(1, SIDE_MAX + 1):
            for ca in range(1, SIDE_MAX + 1):
                for rb in range(1, SIDE_MAX + 1):
                    for cb in range(1, SIDE_MAX + 1):
                        if (ca - ra) * (cb - rb) < 0:
                            continue
                        if not weight_feasible(ra, ca, rb, cb) or not admissible_orders(ra, ca, rb, cb):
                            continue
                        key = shape_key(ra, ca, rb, cb)
                        if key not in seen:
                            seen.add(key)
                            fams.append(key)
        _FAMS5 = fams
    return _FAMS5


def shapes_in_band_v5(lo, hi):
    return [fm for fm in shape_families_v5() if lo <= shape_rate(*_flat(fm)) < hi]


_INV = {}


def element_order_stats(pres, q):
    """Sorted (element order, count) pairs of the group: equal for isomorphic groups, used to keep a
    template parent's isomorphic library copy out of a level-3 draw."""
    k = (pres, q)
    if k not in _INV:
        ns = {}
        exec(group_block(pres, q), ns)
        ops = ns["custom_ops"](q)
        els, mul, e = ops["elements"], ops["mul"], ops["ident"]
        hist = {}
        for g in els:
            if pres[0] == "sg":                    # a permutation: lcm of its cycle lengths
                seen, o = [False] * len(g), 1
                for i0 in range(len(g)):
                    if not seen[i0]:
                        ln, j = 0, i0
                        while not seen[j]:
                            seen[j] = True
                            j = g[j]
                            ln += 1
                        o = o * ln // math.gcd(o, ln)
            else:
                x, o = g, 1
                while x != e:
                    x = mul(x, g)
                    o += 1
            hist[o] = hist.get(o, 0) + 1
        _INV[k] = tuple(sorted(hist.items()))
    return _INV[k]


def _other_group_at(p, q, rng=None):
    """A library group of order q that is not the parent's group (None if there is none). With rng
    None, only whether one exists."""
    pres_list = presentations(q)
    if rng is not None and not pres_list:
        return None
    if q != p["q"]:
        return rng.choice(pres_list) if rng is not None else bool(pres_list)
    if p["pres"] is None:
        return None if rng is not None else False
    twin = library_twin(p["pres"])           # the parent's own group, as a library presentation
    others = [x for x in pres_list if x != twin]
    if rng is None:
        return bool(others)
    return rng.choice(others) if others else None


def draw_child_v5(p, level, band, rng):
    new_seed = rng.randrange(1, 2 ** 31)
    ra, ca = p["A"]
    rb, cb = p["B"]
    if level == 2 and band is not None:
        lo, hi = band
        lo = lo - 0.05                 # one cell downward (measured k exceeds the bound)
        key0 = shape_key(ra, ca, rb, cb)
        cands = [fm for fm in shapes_in_band_v5(lo, hi) if fm != key0
                 and any(family_has_order(p["pres"], qq) for qq in admissible_orders(*_flat(fm)))]
        if not cands:
            return new_seed, None, None, None, "L2 (v5): no other shape in [%g, %g) admits the parent's group family - filling re-rolled only" % (max(lo, 0.0), hi)
        (ra, ca), (rb, cb) = rng.choice(cands)
        if rng.random() < 0.5:                       # A/B roles at random
            (ra, ca), (rb, cb) = (rb, cb), (ra, ca)
        f = shape_factor(ra, ca, rb, cb)
        if p["q"] in admissible_orders(ra, ca, rb, cb):
            return new_seed, (ra, ca, rb, cb), None, p["q"], "L2 (v5): shape -> (%d,%d)x(%d,%d), group kept (order %d admissible, n=%d)" % (ra, ca, rb, cb, p["q"], f * p["q"])
        q = draw_order(ra, ca, rb, cb, rng, admits=lambda qq: family_has_order(p["pres"], qq))
        pres = rng.choice(family_members(p["pres"], q))
        return new_seed, (ra, ca, rb, cb), pres, q, "L2 (v5): shape -> (%d,%d)x(%d,%d), order %d not admissible, same family at order %d (nearest to an LLM block length): %s" % (ra, ca, rb, cb, p["q"], q, group_tag(pres))
    if level == 3:
        def possible(shape):
            return any(_other_group_at(p, qq) for qq in admissible_orders(*shape))
        shape0 = (ra, ca, rb, cb)
        forced = not possible(shape0)
        new_shapes, how = None, ""
        if forced or rng.random() < RND_L3_SHAPE_P:
            key0 = shape_key(*shape0)
            cands = [fm for fm in shape_families_v5() if fm != key0 and possible(_flat(fm))]
            if cands:
                (ra, ca), (rb, cb) = rng.choice(cands)
                if rng.random() < 0.5:
                    (ra, ca), (rb, cb) = (rb, cb), (ra, ca)
                new_shapes = (ra, ca, rb, cb)
                how = ", shapes -> (%d,%d)x(%d,%d)%s" % (ra, ca, rb, cb, " (forced: no other group on the parent's shapes)" if forced else "")
            elif forced:
                return new_seed, None, None, None, "L3 (v5): no other group reachable - filling re-rolled only"
        # one target block length; the nearest order that holds a group other than the parent's
        q = draw_order(ra, ca, rb, cb, rng, admits=lambda qq: _other_group_at(p, qq))
        pres = _other_group_at(p, q, rng)
        return new_seed, new_shapes, pres, q, "L3 (v5): group -> %s q=%d (order nearest to an LLM block length, uniform over the %s other groups of that order)%s" % (group_tag(pres), q, "library" if q != p["q"] else "remaining", how)
    return new_seed, None, None, None, "L1: filling re-rolled (BASE_SEED %d -> %d)" % (p["seed"], new_seed)
