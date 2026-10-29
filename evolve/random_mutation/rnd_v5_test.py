"""Offline checks of the v5 random-mutation rules, no GPU.

  [A] RND_V5 unset: rnd_mutator.py answers byte for byte as rnd_mutator.v4.py for the same generator
      state, over random chains of all levels and cells, for two rule settings.
  [B] RND_V5=1, level 2: every child either keeps the parent's group (its order is admissible on the
      new shapes) or, only when that order is not admissible, moves to another order of the same
      family; or, when no shape of the band admits the family, re-rolls the filling only.
  [C] RND_V5=1, lowest cell: bound-0 shapes, (1,1)x(1,1) among them, are drawn; no negative bound,
      sides <= 8.
  [D] RND_V5=1, level 3: the group always differs from the parent's (library parent: other group;
      template parent at its own order: different element-order statistics); shapes are redrawn at
      the rate RND_L3_SHAPE_P (forced redraws counted apart), from all shapes.
  [E] built children: every slot builds (slots_ok == CM_NSLOTS), n in the window, q as drawn; the
      sample includes (1,1)x(1,1), other bound-0 shapes, level-3 shape changes and template parents
      moved to another group of their own order.
  [F] HTTP server with v5: open lowest cell parsed to (-inf, edge), settings in the start record.
  [G] the structure-description file must match the library exactly (missing, extra, duplicate and
      malformed lines fail); the template-id file must cover every template presentation.
  [H] every template presentation and its library twin (GAP IdGroup) have the same element-order
      statistics and the same commutativity (a necessary condition, checked on all 3,270).
  [I] v5 children carry rewritten CONCEPT / GROUP / SHAPES lines that name the drawn bed.
Run: python3 rnd_v5_test.py --seeds DIR --lib smallgrp_lib.jsonl --ndist llm_n_distribution.json
     --sd smallgrp_sd.tsv --tmpl template_ids.tsv [--build N]
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from openevolve.utils.code_utils import apply_diff  # noqa: E402

EDGES = (0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525)
NSLOTS = 4


def load(path, name, env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def band_of(cell, v5, RM):
    if cell == 0:
        return (RM.OPEN_LO if v5 else 1e-9, 0.125)
    if cell == 9:
        return (0.525, 1.0)
    return (EDGES[cell - 1], EDGES[cell])


def run_child(code):
    env = dict(os.environ, CM_NMIN="100", CM_NMAX="600", CM_WMAX="12", CM_NSLOTS=str(NSLOTS))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], env=env, capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    rows = [json.loads(x) for x in r.stdout.splitlines() if x.strip().startswith("{")]
    assert rows, r.stdout[-500:]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ndist", required=True)
    ap.add_argument("--sd", required=True)
    ap.add_argument("--tmpl", required=True)
    ap.add_argument("--l3p", type=float, default=0.63)
    ap.add_argument("--build", type=int, default=0)
    ap.add_argument("--port", type=int, default=23991)
    a = ap.parse_args()
    base = dict(RND_LIBRARY="smallgrp", RND_LIB_PATH=a.lib, RND_ORDER_RULE="llm_n", RND_N_DIST=a.ndist,
                RND_N_KEY="fom", RND_WCAPS="12", RND_SD_PATH=a.sd, RND_TMPL_PATH=a.tmpl)
    seeds = sorted(glob.glob(os.path.join(a.seeds, "seed_*.py")))
    codes = [open(s).read() for s in seeds]
    assert len(codes) == 8, seeds

    # [A]
    for env in (dict(base, RND_V5=None, RND_L3_SHAPE_P="0.5"),
                dict(base, RND_V5=None, RND_LIBRARY=None, RND_ORDER_RULE=None, RND_WCAPS=None, RND_L3_SHAPE_P="junk")):
        V4 = load(os.path.join(HERE, "rnd_mutator.v4.py"), "rm_v4", env)
        V5 = load(os.path.join(HERE, "rnd_mutator.py"), "rm_v5off", env)
        assert not V5.RND_V5 and V5.RND_L3_SHAPE_P == 0.0
        n = 0
        for chain in range(60):
            r4, r5 = random.Random(chain), random.Random(chain)
            code = codes[chain % 8]
            for step in range(25):
                level = r4.choice([1, 2, 3]); r5.choice([1, 2, 3])
                cell = r4.randrange(10); r5.randrange(10)
                band = band_of(cell, False, V4) if level == 2 else None
                t4, _ = V4.make_response(code, level, band, r4)
                t5, _ = V5.make_response(code, level, band, r5)
                assert t4 == t5, (chain, step, level, band)
                code = apply_diff(code, t4)
                n += 1
        print("[A] v5 off == v4 byte for byte: %d responses (%s)" % (n, env.get("RND_ORDER_RULE") or "top/templates"))

    RM = load(os.path.join(HERE, "rnd_mutator.py"), "rnd_mutator", dict(base, RND_V5="1", RND_L3_SHAPE_P=str(a.l3p)))
    assert RM.RND_V5 and RM.SIDE_MAX == 8 and abs(RM.RND_L3_SHAPE_P - a.l3p) < 1e-12
    rng = random.Random(20261015)
    parents = list(codes)
    for chain in range(80):
        code = codes[chain % 8]
        for step in range(8):
            level = rng.choice([1, 2, 3])
            band = band_of(rng.randrange(10), True, RM) if level == 2 else None
            code = apply_diff(code, RM.make_response(code, level, band, rng)[0])
            parents.append(code)
    kinds = collections.Counter(RM.parse_program(c)["pres"][0] for c in parents)
    print("parent pool: %d programs %s" % (len(parents), dict(kinds)))

    # [B]
    st = collections.Counter()
    for code in parents:
        p0 = RM.parse_program(code)
        for cell in range(10):
            band = band_of(cell, True, RM)
            text, rec = RM.make_response(code, 2, band, rng)
            p1 = RM.parse_program(apply_diff(code, text))
            st["n"] += 1
            if rec["shapes"] is None:
                assert "re-rolled only" in rec["note"] and p1["A"] == p0["A"] and p1["B"] == p0["B"]
                st["reroll_only"] += 1
                continue
            ra, ca, rb, cb = rec["shapes"]
            assert RM.shape_key(ra, ca, rb, cb) != RM.shape_key(*p0["A"], *p0["B"]), rec["note"]
            assert band[0] - 0.05 <= RM.shape_rate(ra, ca, rb, cb) < band[1]
            assert (ca - ra) * (cb - rb) >= 0 and max(ra, ca, rb, cb) <= 8 and RM.weight_feasible(ra, ca, rb, cb)
            adm = RM.admissible_orders(ra, ca, rb, cb)
            if rec["pres"] is None:
                assert p0["q"] in adm and p1["group_text"] == p0["group_text"]
                st["kept"] += 1
            else:
                assert p0["q"] not in adm, rec["note"]
                assert rec["q"] in adm and rec["q"] != p0["q"]
                assert RM.family_key(rec["pres"]) == RM.family_key(p0["pres"]), (p0["pres"], rec["pres"])
                st["order_moved"] += 1
    print("[B] L2 draws %d: group kept %d (%.0f%%), order moved within the family %d, filling only %d"
          % (st["n"], st["kept"], 100.0 * st["kept"] / st["n"], st["order_moved"], st["reroll_only"]))

    # [C]
    allk = RM.shapes_in_band_v5(RM.OPEN_LO, 0.125)
    zero = [f for f in allk if (f[0][1] - f[0][0]) * (f[1][1] - f[1][0]) == 0]
    assert ((1, 1), (1, 1)) in allk and all((f[0][1] - f[0][0]) * (f[1][1] - f[1][0]) >= 0 for f in allk)
    cnt = collections.Counter()
    for i in range(3000):
        text, rec = RM.make_response(parents[i % len(parents)], 2, band_of(0, True, RM), rng)
        if rec["shapes"] is None:
            cnt["none"] += 1
            continue
        ra, ca, rb, cb = rec["shapes"]
        cnt["zero" if (ca - ra) * (cb - rb) == 0 else "pos"] += 1
        cnt["11x11"] += RM.shape_key(ra, ca, rb, cb) == ((1, 1), (1, 1))
    print("[C] lowest cell: %d shape families (%d with bound 0, of all shapes %d); 3000 L2 draws: bound 0 %d, positive %d, (1,1)x(1,1) %d, filling only %d"
          % (len(allk), len(zero), len(RM.shape_families_v5()), cnt["zero"], cnt["pos"], cnt["11x11"], cnt["none"]))

    # [D]
    d = collections.Counter()
    for i in range(5000):
        code = parents[i % len(parents)]
        p0 = RM.parse_program(code)
        text, rec = RM.make_response(code, 3, None, rng)
        p1 = RM.parse_program(apply_diff(code, text))
        d["n"] += 1
        if rec["pres"] is None:
            d["none"] += 1
            continue
        assert p1["pres"][0] == "sg" and p1["pres"] != RM.library_twin(p0["pres"]), rec["note"]
        if p0["pres"][0] != "sg" and p1["q"] == p0["q"]:
            d["template_same_order_other_group"] += 1
        forced = "forced" in rec["note"]
        d["forced"] += forced
        if rec["shapes"] is not None:
            d["shape"] += 1
            ra, ca, rb, cb = rec["shapes"]
            assert RM.shape_key(ra, ca, rb, cb) != RM.shape_key(*p0["A"], *p0["B"])
            assert (ca - ra) * (cb - rb) >= 0 and max(ra, ca, rb, cb) <= 8 and RM.weight_feasible(ra, ca, rb, cb)
            assert (p1["A"], p1["B"]) == ((ra, ca), (rb, cb))
            d["cell_moved"] += RM.shape_rate(ra, ca, rb, cb) // 0.05 != RM.shape_rate(*p0["A"], *p0["B"]) // 0.05
        else:
            assert (p1["A"], p1["B"]) == (p0["A"], p0["B"]) and not forced
        assert rec["q"] in RM.admissible_orders(*p1["A"], *p1["B"])
    rate = (d["shape"] - d["forced"]) / max(1, d["n"] - d["forced"] - d["none"])
    print("[D] L3 draws %d: shapes redrawn %d (%.3f of unforced, target %.2f; %d forced), template parents moved to another group of their order %d, no group reachable %d"
          % (d["n"], d["shape"], rate, a.l3p, d["forced"], d["template_same_order_other_group"], d["none"]))
    assert abs(rate - a.l3p) < 0.03 and d["none"] == 0

    # [E]
    if a.build:
        picks = collections.defaultdict(list)
        tries = 0
        while tries < 40000 and (len(picks) < 5 or min(len(v) for v in picks.values()) < max(1, a.build // 5)):
            tries += 1
            code = parents[tries % len(parents)]
            p0 = RM.parse_program(code)
            if tries % 2:
                text, rec = RM.make_response(code, 2, band_of(0, True, RM), rng)
            else:
                text, rec = RM.make_response(code, 3, None, rng)
            if rec["shapes"] is None and rec["pres"] is None:
                continue
            sh = rec["shapes"] or (*p0["A"], *p0["B"])
            ra, ca, rb, cb = sh
            if RM.shape_key(*sh) == ((1, 1), (1, 1)):
                tag = "(1,1)x(1,1)"
            elif (ca - ra) * (cb - rb) == 0:
                tag = "bound 0"
            elif rec["level"] == 3 and rec["shapes"] is not None:
                tag = "L3 new shapes"
            elif rec["level"] == 3 and p0["pres"][0] != "sg" and rec["q"] == p0["q"]:
                tag = "template same order"
            else:
                tag = "other"
            if len(picks[tag]) < max(1, a.build // 5):
                picks[tag].append((apply_diff(code, text), rec))
        need = max(1, a.build // 5)
        for tag in ("(1,1)x(1,1)", "bound 0", "L3 new shapes", "template same order", "other"):
            assert len(picks[tag]) >= need, ("coverage not reached", tag, len(picks[tag]), need)
        done = 0
        for tag, lst in sorted(picks.items()):
            for child, rec in lst:
                rows = run_child(child)
                for r in rows:
                    assert 100 <= r["n"] < 600, (r, rec["note"])
                    assert r["q"] == (rec["q"] if rec["q"] is not None else RM.parse_program(child)["q"]), (r, rec["note"])
                    assert r["slots_ok"] == NSLOTS, (r, rec["note"])
                done += 1
            print("    built %-20s %d" % (tag, len(lst)))
        print("[E] %d children built, every slot ok (%d draws searched)" % (done, tries))

    # [F]
    import rnd_llm_server as S
    S.RM = RM
    S.STATE["rng"] = random.Random(3)
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), S.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    code = codes[1]
    seen = collections.Counter()
    for k in range(40):
        user = ("# THIS ROUND - HIGHEST PRIORITY\nAssignment level: L2 (LEVEL_2_SHAPE)\n"
                "THIS ROUND: fill the rate cell 0: k/n below 0.1250 (open-ended)\n"
                "# Current Program\n```python\n%s\n```\n" % code)
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % a.port, data=body,
                                     headers={"Content-Type": "application/json"})
        text = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
        assert "TARGET CELL: [0.0000, 0.1250)" in text, text[:300]
        p1 = RM.parse_program(apply_diff(code, text))
        seen["zero" if (p1["A"][1] - p1["A"][0]) * (p1["B"][1] - p1["B"][0]) == 0 else "pos"] += 1
        assert RM.shape_rate(*p1["A"], *p1["B"]) < 0.125
    srv.shutdown()
    print("[F] server, open lowest cell, 40 calls: bound-0 shapes %d, positive %d" % (seen["zero"], seen["pos"]))
    log = os.path.join(tempfile.mkdtemp(), "rndmut.jsonl")
    pr = subprocess.Popen([sys.executable, os.path.join(HERE, "rnd_llm_server.py"), "--port", str(a.port + 1), "--log", log, "--seed", "5"],
                          env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(400):           # binding resolves the host name, which can be slow
        time.sleep(0.2)
        if os.path.exists(log) and os.path.getsize(log):
            break
    pr.terminate()
    first = json.loads(open(log).readline())
    assert first["event"] == "start" and first["settings"]["v5"] is True and abs(first["settings"]["l3_shape_p"] - a.l3p) < 1e-12, first
    print("[F] server start record: %s" % json.dumps(first["settings"]))

    # [G]
    tmp = tempfile.mkdtemp()
    lines = open(a.sd).read().splitlines()
    for name, content in (("missing", lines[:-1]), ("extra", lines + ["300\t1\tC300"]),
                          ("duplicate", lines[:-1] + [lines[0]]), ("malformed", lines[:-1] + ["5 1 C5"])):
        path = os.path.join(tmp, name + ".tsv")
        open(path, "w").write("\n".join(content) + "\n")
        RM._SDPAT, RM.RND_SD_PATH = None, path
        try:
            RM._sd_patterns()
            raise AssertionError("inconsistent file accepted: " + name)
        except RuntimeError as e:
            print("[G] %s line rejected: %s" % (name, e))
    RM._SDPAT, RM.RND_SD_PATH = None, a.sd
    RM._sd_patterns()
    tl = open(a.tmpl).read().splitlines()
    path = os.path.join(tmp, "tmpl_missing.tsv")
    open(path, "w").write("\n".join(tl[:-1]) + "\n")
    RM._TMPL, RM.RND_TMPL_PATH = None, path
    try:
        RM.template_ids()
        raise AssertionError("short template file accepted")
    except RuntimeError as e:
        print("[G] short template file rejected: %s" % e)
    for name, bad_line in (("wrong order", "mc\t28\t14\t2\t13\t3\nmc\t29\t14\t2\t13\t3"),
                           ("id 0", None)):
        lines2 = [l for l in tl if not l.startswith("mc\t28\t14\t2\t13\t")]
        if name == "wrong order":
            lines2.append("mc\t29\t14\t2\t13\t3")
        else:
            lines2.append("mc\t28\t14\t2\t13\t0")
        path = os.path.join(tmp, "tmpl_bad.tsv")
        open(path, "w").write("\n".join(lines2) + "\n")
        RM._TMPL, RM.RND_TMPL_PATH = None, path
        try:
            RM.template_ids()
            raise AssertionError("bad template line accepted: " + name)
        except RuntimeError as e:
            print("[G] template line with %s rejected: %s" % (name, e))
    RM._TMPL, RM.RND_TMPL_PATH = None, a.tmpl
    RM.template_ids()
    try:
        load(os.path.join(HERE, "rnd_mutator.py"), "rm_badcombo", dict(base, RND_V5="1", RND_LIBRARY=None))
        raise AssertionError("RND_V5 without the library accepted")
    except RuntimeError as e:
        print("[G] %s" % e)
    for k, v in base.items():
        os.environ[k] = v

    # [H]
    def commutes(pres, q):
        ns = {}
        exec(RM.group_block(pres, q), ns)
        ops = ns["custom_ops"](q)
        els, mul = ops["elements"], ops["mul"]
        r = random.Random(1)
        return all(mul(x, y) == mul(y, x) for x, y in ((r.choice(els), r.choice(els)) for _ in range(400)))
    bad = checked = skipped = 0
    tm = RM.template_ids()
    for pres, (q, gid) in sorted(tm.items()):
        tw = ("sg", q, gid)
        if gid not in RM._sg_lib().get(q, {}):
            assert q == 256, (pres, tw)   # only order 256 is a sample; the loader enforces the rest
            skipped += 1
            continue
        checked += 1
        if RM.element_order_stats(pres, q) != RM.element_order_stats(tw, q) or commutes(pres, q) != commutes(tw, q):
            bad += 1
            print("    mismatch", pres, tw)
    print("[H] %d template presentations: %d checked against their library twins, %d skipped (order 256, twin not in the sample), %d mismatches" % (len(tm), checked, skipped, bad))
    assert bad == 0

    # [I]
    import re as _re
    for code in parents[8:40]:
        p0 = RM.parse_program(code)
        for lvl in (1, 2, 3):
            text, rec = RM.make_response(code, lvl, band_of(4, True, RM) if lvl == 2 else None, rng)
            child = apply_diff(code, text)
            c = _re.search(r'^CONCEPT = \("(.*)"\)$', child, _re.M).group(1)
            assert c.startswith("Random-mutation arm, not designed:") and ("card level %d" % lvl) in c, c
            p1 = RM.parse_program(child)
            assert ("shapes (%d,%d)x(%d,%d)" % (*p1["A"], *p1["B"])) in c, (c, p1["A"], p1["B"])
            for key in ("GROUP", "SHAPES"):
                assert _re.search(r'^%s = \("Drawn at random by the random-mutation arm' % key, child, _re.M), key
    print("[I] CONCEPT / GROUP / SHAPES rewritten in 96 children")
    print("ALL OK")


if __name__ == "__main__":
    main()
