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
import ast
import hashlib
import json
import os
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
# Local-move variant: RND_LOCAL=1 turns level-1 children
# into local moves around the parent's best measured code (make_response_local, end of this file). With
# the variable unset or anything but "1" nothing in this file changes.
RND_LOCAL = os.environ.get("RND_LOCAL", "") == "1"
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


def make_response(code, level, band, rng, best=None):
    """The full 'LLM response': header lines the engine logs / parses, then
    the SEARCH/REPLACE blocks. Raises ParseError when the parent cannot be
    read (the server then answers with no diff and the round fails cleanly).
    best (RND_LOCAL=1 only, ignored otherwise): the parent's best measured
    slot as the server read it from the prompt, see make_response_local."""
    if RND_LOCAL:
        return make_response_local(code, level, band, rng, best)
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


# --------------------------------------------------------------------------
# local-move variant (RND_LOCAL=1)
# --------------------------------------------------------------------------
LOCAL_K = (1, 2, 3)           # edits per slot, uniform
LOCAL_TRIES = 1000            # redrawn edits allowed per slot before the slot is voided
_END_MARK = "# EVOLVE-BLOCK-END"
_RE_LOCAL_BASE = re.compile(r"^LOCAL_BASE = [^\n]*$", re.M)
_RE_LOCAL_MARK = re.compile(r"^_rndloc_own_NEW_CTX = NEW_CTX$", re.M)
_RE_GTYPE = re.compile(r'^GTYPE = "([^"\n]*)"', re.M)

# Inserted once, right before the EVOLVE-BLOCK-END marker line, by the first level-1 child that
# gets a base. It runs after the program's own NEW_CTX / CELL are defined and keeps them; the
# scaffold calls NEW_CTX / CELL by name at fill time, so the definitions below are the ones used.
# Every rule it enforces is also re-checked by the scaffold's _fill_slot.
LOCAL_WRAPPER = '''\
# ---- local moves (random-mutation arm, RND_LOCAL=1; inserted once by rnd_mutator.py) ----
# LOCAL_BASE: the cells of the parent's best measured code in this program's element encoding,
# or None. With a base of shapes A_SHAPE x B_SHAPE every slot is that code after k local edits,
# k uniform in LOCAL_K. One edit: a cell uniform over all cells of A and B, an operation uniform
# over replace / add / remove, the element removed or replaced uniform over the cell, the new
# element uniform over the group elements not in the cell. An edit that breaks a slot rule
# (0..CELL_MAX distinct elements per cell, row weight <= W_MAX, no all-zero protograph row or
# column) or does not take the slot one edit further from the base is redrawn; a slot whose drawn
# number of edits cannot be completed within LOCAL_TRIES redraws keeps the edits it has (k_done in
# LOCAL_LOG), and only a slot with no legal edit at all is voided. A replacement
# that only renames qubits and checks is redrawn, and a slot whose code is the base with renamed
# qubits and checks is redrawn as a whole (_rndloc_renamed). All draws use the slot's rng.
# Without a base the program's own NEW_CTX / CELL, kept here, fill every slot.
LOCAL_BASE = None
LOCAL_K = @K@
LOCAL_TRIES = @TRIES@
LOCAL_RENAME_BUDGET = 200000
LOCAL_LOG = []
_rndloc_own_NEW_CTX = NEW_CTX
_rndloc_own_CELL = CELL
_rndloc_tabs = {}


class _RndlocCtx(dict):
    pass


def _rndloc_tables(G):
    t = _rndloc_tabs.get(id(G))
    if t is None:
        els = G.elements
        t = ([[G.idx(G.mul(a, b)) for b in els] for a in els], [G.idx(G.inv(a)) for a in els])
        _rndloc_tabs[id(G)] = t
    return t


def _rndloc_renamed(G, M1, M2):
    """True when M2[r][c] = {x_r * g * y_c : g in M1[r][c]} for all cells, for some group elements
    x_r (one per protograph row) and y_c (one per protograph column). Under the lift of
    lifted_matrices this holds exactly when M2 is M1 up to a renaming of qubits and checks
    within their blocks, so A or B can be replaced by M2 without changing the code. None when the
    search takes more than LOCAL_RENAME_BUDGET cell checks."""
    mul, inv = _rndloc_tables(G)
    R, C = len(M1), len(M1[0])
    S1 = [[[G.idx(g) for g in cell] for cell in row] for row in M1]
    S2 = [[frozenset(G.idx(g) for g in cell) for cell in row] for row in M2]
    if any(len(set(S1[r][c])) != len(S2[r][c]) for r in range(R) for c in range(C)):
        return False
    nb = [[] for _ in range(R + C)]          # vertices: rows 0..R-1, columns R..R+C-1
    for r in range(R):
        for c in range(C):
            if S1[r][c]:
                nb[r].append(R + c)
                nb[R + c].append(r)
    val = [None] * (R + C)
    work = [0]

    def holds(r, c):
        work[0] += 1
        x, y = val[r], val[R + c]
        return frozenset(mul[mul[x][g]][y] for g in S1[r][c]) == S2[r][c]

    def options(v):
        for w in nb[v]:
            if val[w] is not None:
                if v < R:                    # row v from column w: x = b * (g0 * y)^-1
                    g0, y = S1[v][w - R][0], val[w]
                    return [mul[b][inv[mul[g0][y]]] for b in S2[v][w - R]]
                g0, x = S1[w][v - R][0], val[w]   # column v from row w: y = (x * g0)^-1 * b
                return [mul[inv[mul[x][g0]]][b] for b in S2[w][v - R]]
        return range(len(inv))

    def solve(order, i):
        if i == len(order):
            return True
        if work[0] > LOCAL_RENAME_BUDGET:
            return None
        v = order[i]
        for a in options(v):
            val[v] = a
            if all(val[w] is None or (holds(v, w - R) if v < R else holds(w, v - R)) for w in nb[v]):
                got = solve(order, i + 1)
                if got is not False:
                    return got
        val[v] = None
        return False

    seen = set()
    for v0 in range(R + C):
        if v0 in seen or not nb[v0]:
            continue
        order = [v0]
        seen.add(v0)
        for v in order:
            for w in nb[v]:
                if w not in seen:
                    seen.add(w)
                    order.append(w)
        got = solve(order, 0)
        if got is not True:
            return got
    return True


def _rndloc_key(x):
    return tuple(_rndloc_key(y) for y in x) if isinstance(x, (list, tuple)) else x


def _rndloc_dist(G, base_cell, cell):
    b = set(G._key(g) for g in base_cell)
    c = set(G._key(g) for g in cell)
    return max(len(b - c), len(c - b))


def _rndloc_base(G):
    """LOCAL_BASE with G's own element objects; None without a base or when its grid shapes are
    not A_SHAPE / B_SHAPE; an element outside G is an error (every slot is voided)."""
    if LOCAL_BASE is None:
        return None
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = LOCAL_BASE.get(side)
        if not isinstance(grid, list) or len(grid) != rows or any(len(r) != cols for r in grid):
            return None
    els = {_rndloc_key(g): g for g in G.elements}
    out = {}
    for side in ("A", "B"):
        out[side] = []
        for row in LOCAL_BASE[side]:
            out[side].append([])
            for cell in row:
                miss = [x for x in cell if _rndloc_key(x) not in els]
                if miss:
                    raise ValueError("LOCAL_BASE element %r is not in the group" % (miss[0],))
                out[side][-1].append([els[_rndloc_key(x)] for x in cell])
    return out


def NEW_CTX(rng, G, idx):
    base = _rndloc_base(G)
    if base is None:
        return _rndloc_own_NEW_CTX(rng, G, idx)
    where = [(s, i, j) for s in ("A", "B") for i in range(len(base[s])) for j in range(len(base[s][i]))]
    k = rng.choice(LOCAL_K)
    redrawn, unchecked = {}, 0
    while True:
        cells = {s: [[list(c) for c in row] for row in base[s]] for s in ("A", "B")}
        sup = {s: [[len(c) for c in row] for row in cells[s]] for s in ("A", "B")}
        ops, at = [], []
        while len(ops) < k:
            if sum(redrawn.values()) >= LOCAL_TRIES:
                if ops:
                    break                # fewer edits than drawn: the slot keeps the edits it has (k_done < k)
                raise ValueError("local move: no legal edit in %d draws" % LOCAL_TRIES)
            s, i, j = rng.choice(where)
            op = rng.choice(("replace", "add", "remove"))
            cell = cells[s][i][j]
            keys = set(G._key(g) for g in cell)
            free = [g for g in G.elements if G._key(g) not in keys]
            new, why = list(cell), None
            if op != "add" and not cell:
                why = "empty cell"
            elif op == "add" and len(cell) >= CELL_MAX:
                why = "full cell"
            elif op != "remove" and not free:
                why = "no free element"
            elif op == "replace":
                new[rng.randrange(len(new))] = rng.choice(free)
            elif op == "add":
                new.append(rng.choice(free))
            else:
                del new[rng.randrange(len(new))]
            if why is None:
                sup[s][i][j] = len(new)
                if _grid_weights(sup["A"], sup["B"]) > W_MAX:
                    why = "weight cap"
                elif not _no_dead_lines(sup["A"], sup["B"]):
                    why = "dead line"
                elif _rndloc_dist(G, base[s][i][j], new) != _rndloc_dist(G, base[s][i][j], cell) + 1:
                    why = "not one edit further"
                elif op == "replace":
                    trial = [list(row) for row in cells[s]]
                    trial[i][j] = new
                    same = _rndloc_renamed(G, cells[s], trial)
                    if same:
                        why = "renames qubits only"
                    elif same is None:
                        unchecked += 1
                if why is not None:
                    sup[s][i][j] = len(cell)
            if why is None:
                cells[s][i][j] = new
                ops.append(op)
                at.append([s, i, j])
            else:
                redrawn[op + ": " + why] = redrawn.get(op + ": " + why, 0) + 1
        same = [_rndloc_renamed(G, base[s], cells[s]) for s in ("A", "B")]
        if None in same:
            unchecked += 1
        if not all(same):
            break
        redrawn["slot: base with renamed qubits"] = redrawn.get("slot: base with renamed qubits", 0) + 1
        if sum(redrawn.values()) >= LOCAL_TRIES:
            raise ValueError("local move: no legal edit in %d draws" % LOCAL_TRIES)
    rec = {"idx": idx, "k": k, "k_done": len(ops), "ops": ops, "at": at, "redrawn": redrawn}
    if unchecked:
        rec["rename_check_over_budget"] = unchecked
    LOCAL_LOG.append(rec)
    return _RndlocCtx(cells=cells)


def CELL(rng, G, side, i, j, ctx):
    if isinstance(ctx, _RndlocCtx):
        return list(ctx["cells"][side][i][j])
    return _rndloc_own_CELL(rng, G, side, i, j, ctx)'''.replace("@K@", repr(LOCAL_K)).replace("@TRIES@", str(LOCAL_TRIES))


def _deep_key(x):
    return tuple(_deep_key(y) for y in x) if isinstance(x, (list, tuple)) else x


def _plain(x):
    """Cell elements must be ints, strings or (nested) lists of them, so that their JSON text is
    also a Python literal (JSON true / false / null / floats are refused)."""
    if isinstance(x, bool) or not isinstance(x, (int, str, list)):
        return False
    return all(_plain(y) for y in x) if isinstance(x, list) else True


def _grid_weights(sa, sb):
    """The scaffold's _grid_weights: max HX row weight (A row i + B column j) and HZ row weight
    (B row r + A column c) of the support grids."""
    wx = max(sum(sa[i]) + sum(r[j] for r in sb) for i in range(len(sa)) for j in range(len(sb[0])))
    wz = max(sum(sb[r]) + sum(row[c] for row in sa) for r in range(len(sb)) for c in range(len(sa[0])))
    return max(wx, wz)


def _no_dead_lines(sa, sb):
    """The scaffold's _no_dead_lines: every protograph row and column of both grids is nonzero."""
    return all(all(sum(row) > 0 for row in g) and all(sum(r[c] for r in g) > 0 for c in range(len(g[0])))
               for g in (sa, sb))


_GROUP_KEYS = {}


def _group_keys(group_text, q):
    """Keys of the elements of the parent's own CUSTOM_GROUP at order q, built the way the scaffold
    builds them: the CUSTOM_GROUP assignment gives the code string, exec of the string defines
    valid_q / custom_ops, custom_ops(q)["elements"] lists the elements. Cached per (text, q)."""
    ck = (hashlib.sha1(group_text.encode()).hexdigest(), q)
    if ck not in _GROUP_KEYS:
        if len(_GROUP_KEYS) > 256:
            _GROUP_KEYS.clear()
        ns0, ns = {}, {}
        exec(group_text, ns0)
        exec(ns0["CUSTOM_GROUP"], ns)
        if not ns["valid_q"](q):
            raise ValueError("valid_q rejects q=%d" % q)
        els = list(ns["custom_ops"](q)["elements"])
        keys = set(_deep_key(g) for g in els)
        if len(els) != q or len(keys) != q:
            raise ValueError("custom_ops(%d) lists %d elements, %d distinct" % (q, len(els), len(keys)))
        _GROUP_KEYS[ck] = keys
    return _GROUP_KEYS[ck]


def local_base_literal(p, code, best, cap):
    """(literal, None) when best carries cells a level-1 child of this parent can start from, else
    (None, reason). The checks are the scaffold's slot rules on the parent's own bed: grid shapes
    A_SHAPE / B_SHAPE, 0..CELL_MAX distinct elements per cell, every element in the parent's group,
    row weight <= cap (the child's W_MAX), no all-zero protograph row or column."""
    if not isinstance(best, dict) or "cells" not in best:
        return None, (best or {}).get("why") or "no best measured slot was passed"
    cells = best["cells"]
    if not isinstance(cells, dict) or set(cells) != {"A", "B"}:
        return None, "the best cells are not an A/B pair"
    sup = {}
    for side in ("A", "B"):
        rows, cols = p[side]
        grid = cells[side]
        if (not isinstance(grid, list) or len(grid) != rows
                or any(not isinstance(r, list) or len(r) != cols for r in grid)):
            return None, "the best cells do not have the parent's %s shape (%d, %d)" % (side, rows, cols)
        for row in grid:
            for cell in row:
                if not isinstance(cell, list) or len(cell) > CELL_MAX:
                    return None, "a best cell is not a list of at most %d elements" % CELL_MAX
                if not all(_plain(x) for x in cell):
                    return None, "a best cell holds an element that is not an int, a string or a list"
                if len(set(_deep_key(x) for x in cell)) != len(cell):
                    return None, "a best cell repeats an element"
        sup[side] = [[len(c) for c in row] for row in grid]
    if not _no_dead_lines(sup["A"], sup["B"]):
        return None, "the best cells have an all-zero protograph row or column"
    w = _grid_weights(sup["A"], sup["B"])
    if w > cap:
        return None, "the best cells have row weight %d above the child's cap %d" % (w, cap)
    gm = _RE_GTYPE.search(code)
    if not gm or gm.group(1) != "custom" or not p.get("q"):
        return None, "the parent's group is not a CUSTOM_GROUP with a fixed order"
    try:
        keys = _group_keys(p["group_text"], p["q"])
    except Exception as e:
        return None, "the parent's group could not be built (%s: %s)" % (type(e).__name__, str(e)[:80])
    for side in ("A", "B"):
        for row in cells[side]:
            for cell in row:
                for x in cell:
                    if _deep_key(x) not in keys:
                        return None, "the best cells hold %s, not an element of the parent's group" % json.dumps(x)[:40]
    lit = json.dumps(cells, separators=(",", ":"))
    if ast.literal_eval(lit) != cells:
        return None, "the best cells do not read back as a Python literal"
    return lit, None


_RE_WMAX_CAP = re.compile(r'^W_MAX = min\(int\(os\.environ\.get\("CM_WMAX", "\d+"\)\), (\d+)\)')


def _parent_cap(p, env_cap):
    """The row-weight cap of the parent: the second argument of its 'W_MAX = min(..., c)' line (a
    random-arm child), else the environment cap (a seed's 'W_MAX = int(os.environ.get(...))')."""
    m = _RE_WMAX_CAP.match(p.get("wmax_line") or "")
    return int(m.group(1)) if m else env_cap


def make_response_local(code, level, band, rng, best=None):
    """RND_LOCAL=1. The draws, blocks and header of make_response (v5 when RND_V5=1), with three differences:
    the family name starts with "rndloc_"; a level-1 child whose parent has usable best cells
    (best: {"cells", "idx", "score"} from the server, or {"why"}) gets LOCAL_BASE = those cells,
    with LOCAL_WRAPPER inserted when the parent has none; every other child gets LOCAL_BASE = None
    when the parent carries a base (a parent without the wrapper keeps none)."""
    p = parse_program(code)
    lbm = _RE_LOCAL_BASE.search(code)
    has_wrapper = bool(_RE_LOCAL_MARK.search(code))
    if has_wrapper != bool(lbm):
        raise ParseError("local-move wrapper and LOCAL_BASE line out of step")
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
    lightest = max(ca + rb, cb + ra)
    caps = [w for w in W_CAPS if w >= lightest] or [W_MAX]
    wcap = drawn = rng.choice(caps)
    # the base: only on a level-1 card. The child takes the drawn cap when the base fits under it;
    # when the drawn cap is below the base's weight (RND_WCAPS with several values, e.g. 9-12 in the
    # CLER line) the child keeps its parent's cap, under which the base was built.
    # The draw itself is unchanged, so the rng stream is too (the same draws as make_response).
    lit, why = None, None
    env_cap = int(os.environ.get("CM_WMAX", str(W_MAX)))
    if level == 1:
        lit, why = local_base_literal(p, code, best, min(env_cap, drawn))
        if lit is None:
            pcap = _parent_cap(p, env_cap)
            if pcap != drawn:
                lit2, why2 = local_base_literal(p, code, best, min(env_cap, pcap))
                if lit2 is not None:
                    lit, why, wcap = lit2, None, pcap
        if lit is not None and not has_wrapper and code.split("\n").count(_END_MARK) != 1:
            lit, why, wcap = None, "the parent has no unique '%s' line to insert the wrapper before" % _END_MARK, drawn
    if p.get("wmax_line"):
        blocks.append(_diff_block(
            p["wmax_line"],
            ('W_MAX = min(int(os.environ.get("CM_WMAX", "%d")), %d)  # row-weight cap kept from the parent '
             '(local move; caps of the random arm: %s)' if lit is not None and wcap != drawn else
             'W_MAX = min(int(os.environ.get("CM_WMAX", "%d")), %d)  # row-weight cap drawn from %s (random arm)')
            % (W_MAX, wcap, list(W_CAPS))))
        changes.append("weight cap %d" % wcap)
    fam = ("rndloc_%s_%d-%dx%d-%d_w%d" if RND_V5 else "rndloc_%s_%d%dx%d%d_w%d") % (tag, ra, ca, rb, cb, wcap)
    if RND_V5:
        qq = q if q is not None else p["q"]
        for key, text in (("CONCEPT", "Random-mutation arm (local moves), not designed: %s of order %s, shapes (%d,%d)x(%d,%d), "
                                      "row-weight cap %d; this child was drawn at card level %d." % (tag, qq, ra, ca, rb, cb, wcap, level)),
                          ("GROUP", "Drawn at random by the random-mutation arm (rnd_mutator.py v5, local moves); no reason."),
                          ("SHAPES", "Drawn at random by the random-mutation arm (rnd_mutator.py v5, local moves); no reason.")):
            m = re.search(r'^%s = \(".*"\)$' % key, code, re.M)
            if m:
                blocks.append(_diff_block(m.group(0), '%s = ("%s")' % (key, text)))
    if p["family_line"]:
        blocks.append(_diff_block(p["family_line"], 'FAMILY_NAME = "%s"' % fam))
    if lit is not None:
        line = "LOCAL_BASE = " + lit
        if has_wrapper:
            blocks.append(_diff_block(lbm.group(0), line))
        else:
            blocks.append(_diff_block(_END_MARK, LOCAL_WRAPPER.replace("LOCAL_BASE = None", line, 1) + "\n" + _END_MARK))
        changes[0] = "filling (local edits of the parent's best code)"
        note = ("L1: local moves - every slot is the parent's best measured code (slot %s, score %s, "
                "%d of %d cells non-empty) after k in %s random edits; BASE_SEED %d -> %d"
                % (best.get("idx"), best.get("score"),
                   sum(1 for s in ("A", "B") for row in best["cells"][s] for c in row if c),
                   ra * ca + rb * cb, list(LOCAL_K), p["seed"], new_seed))
    else:
        if has_wrapper and lbm.group(0) != "LOCAL_BASE = None":
            blocks.append(_diff_block(lbm.group(0), "LOCAL_BASE = None"))
        if level == 1:
            changes[0] = "filling (re-rolled: local-move fallback)"
            note = "L1: local-move fallback (%s) - %s" % (why, note)
    if wcap == drawn:
        note = "%s; row-weight cap %d (drawn from %s, lightest rows %d)" % (note, wcap, list(caps), lightest)
    else:
        note = ("%s; row-weight cap %d (kept from the parent: the drawn cap %d from %s is below the base's weight; "
                "lightest rows %d)" % (note, wcap, drawn, list(caps), lightest))
    head = ["CHANGES: %s" % ", ".join(changes)]
    if band is not None:
        head.append("TARGET CELL: [%.4f, %.4f)" % ((max(band[0], 0.0), band[1]) if RND_V5 else band))
    head.append("REASON: random-mutation arm (local moves) - %s" % note)
    head.append("CONCEPT: random draw at card level L%d%s; %s, shapes (%d,%d)x(%d,%d), q=%s" % (
        level, ", local edits of the parent's best code" if lit is not None else "",
        tag, ra, ca, rb, cb, q if q is not None else p["q"]))
    head.append("WHY: no reason - this arm is the random control")
    head.append("DIFFERS FROM PARENT: %s" % note)
    return "\n".join(head) + "\n\n" + "\n".join(blocks), dict(
        level=level, band=band, note=note, seed=new_seed, shapes=shapes,
        pres=pres, q=q, family=fam, parent_family=p["family"], wcap=wcap,
        wcap_drawn=drawn, wcap_kept=wcap != drawn,
        local=True, local_base=lit is not None, local_why=why,
        wrapper_inserted=lit is not None and not has_wrapper,
        base_idx=(best or {}).get("idx") if lit is not None else None,
        base_score=(best or {}).get("score") if lit is not None else None,
        base_sha1=hashlib.sha1(lit.encode()).hexdigest()[:16] if lit is not None else None,
        parent_sha1=hashlib.sha1(code.encode()).hexdigest()[:16])
