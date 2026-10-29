"""A local OpenAI-compatible endpoint that answers every chat completion with a
random mutation of the prompt's current program (rnd_mutator.py).

Point the engine's llm.api_base at it (http://127.0.0.1:PORT/v1); nothing
else in the engine changes. The server reads three things out of the user
message the engine sends - the dealt card level ("Assignment level: Ln"), the
named rate cell on L2 rounds ("THIS ROUND: fill the rate cell c: k/n in
[lo, hi)") and the parent program (the fenced block under "# Current
Program") - and returns the SEARCH/REPLACE diff of a random child.

  python rnd_llm_server.py --port 23456 --log /path/rndmut.jsonl [--seed N]

One random.Random for the life of the process, behind a lock: the engine
calls from up to parallel_evaluations threads at once.
"""
import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rnd_mutator as RM

_RE_LEVEL = re.compile(r"Assignment\s+level:\s*L([1-9])\b", re.I)
_RE_BAND = re.compile(r"THIS ROUND: fill the rate cell (\d+): k/n in \[([0-9.]+), ([0-9.]+)\)")
# The two end cells are worded open-ended by the engine (prompt/sampler.py):
# "cell 0: k/n below 0.1250" / "cell 9: k/n above 0.5250".
_RE_BAND_LO = re.compile(r"THIS ROUND: fill the rate cell (\d+): k/n below ([0-9]+(?:\.[0-9]+)?)")
_RE_BAND_HI = re.compile(r"THIS ROUND: fill the rate cell (\d+): k/n above ([0-9]+(?:\.[0-9]+)?)")
_RE_PROGRAM = re.compile(r"# Current Program\n```python\n(.*?)\n```", re.S)

# RND_LOCAL=1: the current program's evaluation table is the "- artifacts.slot_table:" item of the
# "# Current Program Information" section (prompt/sampler.py _format_metrics prints it in full, no
# length cap); it ends at the next "- " item ("- artifacts.run_state:", whose FRONTIER lists the
# cells of OTHER programs) or at the next "# " heading.
_RE_CUR_INFO = re.compile(r"^# Current Program Information\n(.*?)(?=^# |\Z)", re.S | re.M)
_TABLE_ITEM = "- artifacts.slot_table:\n"
# score column of the evaluation table: kd30s^2/n or kd_circ2/n (distance lines), s (CLER line: the
# slot score printed with two decimals; the CLER ledger sorts the measured slots by the unrounded value)
_SCORE_COLS = ("kd30s^2/n", "kd_circ2/n", "s")
STATE = {"rng": None, "lock": threading.Lock(), "log": None, "n": 0, "token": None}


def _user_text(body):
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):          # content parts
                return "".join(part.get("text", "") for part in c if isinstance(part, dict))
            return c or ""
    return ""


def _num(tok):
    try:
        return float(tok)
    except ValueError:
        return 0.0                       # "None" / "-": the ledger sorts these as 0 too


def _cells_of(text):
    """{"A": ..., "B": ...} from a table line "cells A=<json> B=<json>", None if malformed."""
    dec = json.JSONDecoder()
    try:
        a, k = dec.raw_decode(text, len("cells A="))
        if not text.startswith(" B=", k):
            return None
        b, k = dec.raw_decode(text, k + len(" B="))
    except ValueError:
        return None
    return {"A": a, "B": b} if not text[k:].strip() else None


def parent_best(user_text):
    """The current program's best measured code, read from its evaluation table in the prompt: the
    measured row with the highest score, the first such row on a tie. These are the cells the
    adapter records as best_cells in the ledger (oe_evaluator_circmean_v3._ledger_write sorts the
    measured slots by -score, stably, in the order the table lists them). Returns
    {"cells", "idx", "score", "n_measured"} or {"why": reason}."""
    m = _RE_CUR_INFO.search(user_text)
    if not m:
        return {"why": "the prompt has no 'Current Program Information' section"}
    sec = m.group(1)
    i = sec.find(_TABLE_ITEM)
    if i < 0:
        return {"why": "the current program has no evaluation table in the prompt"}
    lines = []
    for line in sec[i + len(_TABLE_ITEM):].split("\n"):
        if line.startswith("- "):
            break
        lines.append(line)
    hdr = lines[0].split() if lines else []
    if len(hdr) < 3 or hdr[0] != "idx" or hdr[-1] != "status":
        return {"why": "the evaluation table header is not recognised"}
    sc = next((hdr.index(c) for c in _SCORE_COLS if c in hdr), None)
    if sc is None:
        return {"why": "the evaluation table has no score column"}
    rows, cur = [], None
    for line in lines[1:]:
        parts = line.split(None, len(hdr) - 1)
        if not line.startswith("     ") and len(parts) == len(hdr) and (parts[0].isdigit() or parts[0] == "?"):
            cur = {"idx": int(parts[0]) if parts[0].isdigit() else None, "status": parts[-1].strip(),
                   "score": _num(parts[sc]), "cells": None}
            rows.append(cur)
        elif line.strip().startswith("cells A=") and cur is not None and cur["cells"] is None:
            cur["cells"] = _cells_of(line.strip())
    meas = [r for r in rows if r["status"] == "measured"]
    if not meas:
        return {"why": "the current program has no measured slot"}
    top = max(r["score"] for r in meas)
    b = next(r for r in meas if r["score"] == top)
    if b["cells"] is None:
        return {"why": "the best measured slot (%s) shows no cells" % b["idx"]}
    return {"cells": b["cells"], "idx": b["idx"], "score": b["score"], "n_measured": len(meas)}


def answer(user_text):
    """The response text plus an audit record."""
    m = _RE_LEVEL.search(user_text)
    level = int(m.group(1)) if m else 1
    m = _RE_BAND.search(user_text)
    band = (float(m.group(2)), float(m.group(3))) if m else None
    cell = int(m.group(1)) if m else None
    if m is None:
        m = _RE_BAND_LO.search(user_text)
        if m:   # open low end: every shape whose rank-count rate is positive and below the edge
            # (v5: every shape below the edge, rank-count bound 0 included)
            band, cell = (RM.OPEN_LO if RM.RND_V5 else 1e-9, float(m.group(2))), int(m.group(1))
        else:
            m = _RE_BAND_HI.search(user_text)
            if m:   # open high end
                band, cell = (float(m.group(2)), 1.0), int(m.group(1))
    m = _RE_PROGRAM.search(user_text)
    if not m:
        return ("REASON: random-mutation arm could not find the current program in the prompt\n",
                dict(error="no current program", level=level, cell=cell))
    code = m.group(1)
    best = parent_best(user_text) if RM.RND_LOCAL else None
    with STATE["lock"]:
        try:
            if RM.RND_LOCAL:
                text, rec = RM.make_response(code, level, band, STATE["rng"], best=best)
            else:
                text, rec = RM.make_response(code, level, band, STATE["rng"])
        except RM.ParseError as e:
            return ("REASON: random-mutation arm could not parse the parent (%s)\n" % e,
                    dict(error=str(e), level=level, cell=cell))
    rec["cell"] = cell
    return text, rec


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):     # keep the job log quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = {}
        text, rec = answer(_user_text(body))
        with STATE["lock"]:
            STATE["n"] += 1
            k = STATE["n"]
            if STATE["log"]:
                rec.update(i=k, t=time.time(), path=self.path, model=body.get("model"))
                STATE["log"].write(json.dumps(rec) + "\n")
                STATE["log"].flush()
        out = {
            "id": "rndmut-%d" % k,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or "random-mutator",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(raw) // 4, "completion_tokens": len(text) // 4,
                      "total_tokens": (len(raw) + len(text)) // 4},
        }
        data = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # the token and pid let the controller check that this is its own server
        data = json.dumps({"ok": True, "served": STATE["n"], "token": STATE["token"],
                           "pid": __import__("os").getpid()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--log", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--port-file", default=None, help="write the bound port here (use with --port 0)")
    ap.add_argument("--token", default=None, help="returned by GET /, so the caller can identify its server")
    a = ap.parse_args()
    STATE["token"] = a.token
    seed = a.seed if a.seed is not None else int(time.time() * 1000) % (2 ** 31)
    STATE["rng"] = random.Random(seed)
    settings = dict(v5=RM.RND_V5, library=RM.RND_LIBRARY, order_rule=RM.RND_ORDER_RULE, n_key=RM.RND_N_KEY,
                    wcaps=list(RM.W_CAPS), rnd_local=RM.RND_LOCAL,
                    local_k=list(RM.LOCAL_K) if RM.RND_LOCAL else None, local_tries=RM.LOCAL_TRIES if RM.RND_LOCAL else None,
                    cm_wmax=__import__("os").environ.get("CM_WMAX"),
                    mutator_sha1=__import__("hashlib").sha1(open(RM.__file__, "rb").read()).hexdigest()[:16])
    if RM.RND_V5:
        # v5: load the library, the structure-description file and the template ids now, so that a
        # missing or inconsistent file stops the job here instead of failing level-2/3 rounds later
        RM._sd_patterns()
        RM.template_ids()
        settings.update(sd_path=RM.RND_SD_PATH, tmpl_path=RM.RND_TMPL_PATH, l3_shape_p=RM.RND_L3_SHAPE_P,
                        shape_families=len(RM.shape_families_v5()))
    # bind first: the start record is written only by a server that owns its port
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    port = srv.server_address[1]
    settings.update(pid=__import__("os").getpid(), token=a.token)
    if a.log:
        STATE["log"] = open(a.log, "a")
        STATE["log"].write(json.dumps(dict(event="start", seed=seed, port=port, t=time.time(), settings=settings)) + "\n")
        STATE["log"].flush()
    print("rnd_llm_server settings: %s" % json.dumps(settings), flush=True)
    print("rnd_llm_server on %s:%d seed=%d" % (a.host, port, seed), flush=True)
    if a.port_file:                       # written last and atomically: its presence means "bound"
        tmp = a.port_file + ".tmp"
        with open(tmp, "w") as f:
            f.write("%d\n" % port)
        __import__("os").replace(tmp, a.port_file)
    srv.serve_forever()


if __name__ == "__main__":
    main()
