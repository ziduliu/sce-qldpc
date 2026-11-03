"""Circuit-level pL sampler: GPU first-leg relay-BP screen plus a CPU decoder
cascade (exact_cascade, mode PLC_MODE), in both memory bases.

Chunk c of PL_CHUNK shots comes from a detector sampler seeded 1000 + PL_CHUNK*c,
so any partition of the chunk ids (process pool, Slurm shards) gives the same
integer counts. Sampling stops at PL_MAX_ERRORS failed shots (shots with any
logical error) or PL_SHOTS shots, whichever comes first; chunks still in flight
at the cap are counted.

  PL_P=0.001 PL_ROUNDS=6 PL_SHOTS=20000 ASC_SCHED=... \
      python pl_sample.py codes.json out.json
  python pl_sample.py --collect out.json codes.json   # merge the part files of a sharded run
"""
import json
import math
import os
import sys
import threading
import time

import numpy as np

ACC = "/u/zdliu/lp/circmean/cpu_accel_20260911"
CIRCMEAN = "/u/zdliu/lp/circmean"
# directory of the exact_cascade module with the gpu2 mode (GPU stage-1 and
# stage-2 relay)
PLSTAGE2 = os.environ.get("PL_STAGE2_DIR", "/u/zdliu/lp/plstage2_20260912")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, CIRCMEAN, ACC, PLSTAGE2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ref   = exact_cascade's re-implementation of it, decoders built per call.
# hoist = ref with the stage-3 decoder taken from a process-wide cache.
# gpu2  = hoist with relay stage 2 (and with PLG_STAGE1=gpu stage 1) on the GPU.
MODE = os.environ.get("PLC_MODE", "gpu2")   # the paper's setting (pl_score.py)
# optional path prefix for a per-chunk record of the counts
CHUNKLOG = os.environ.get("PLC_CHUNKLOG", "")

P = float(os.environ.get("PL_P", "0.002"))
ROUNDS = int(os.environ.get("PL_ROUNDS", "6"))
SHOTS = int(os.environ.get("PL_SHOTS", "20000"))
# stopping rule on failed shots; 0 disables it
MAX_ERRORS = int(os.environ.get("PL_MAX_ERRORS", "100"))
CHUNK = int(os.environ.get("PL_CHUNK", "250"))
BASES = os.environ.get("PL_BASES", "Z,X").split(",")
GPU_IMPL = os.environ.get("PL_GPU_IMPL", "padded")

# How many reference chunks ride in one GPU call. 0 asks for an automatic
# choice from the free memory on this device and the DEM's edge count.
GROUP = int(os.environ.get("PLF_GROUP", "0"))
GROUP_CAP = int(os.environ.get("PLF_GROUP_CAP", "16"))
MEM_FRAC = float(os.environ.get("PLF_MEM_FRAC", "0.45"))
PREFETCH = os.environ.get("PLF_PREFETCH", "1") == "1"
DETAILS = os.environ.get("PLF_DETAILS", "") == "1"
PROGRESS = int(os.environ.get("PLF_PROGRESS", "0"))
# -1 sizes the pool from the allocation; 0 runs the cascade inline.
WORKERS = int(os.environ.get("PLF_WORKERS", "-1"))

# PLF_ONESAMPLER=1 draws all shots from one sampler seeded PL_SEED in PL_BATCH
# pieces instead of the seeded chunks (for timing only; cannot be sharded).
ONESAMPLER = os.environ.get("PLF_ONESAMPLER", "") == "1"
# block-level failure map from the scored shots (pl_attrib.FrameAttrib)
ATTRIB = os.environ.get("PLF_ATTRIB", "1") == "1"
BATCH = int(os.environ.get("PL_BATCH", "4000"))
SEED0 = int(os.environ.get("PL_SEED", "20260911"))


def worker_count():
    if WORKERS >= 0:
        return WORKERS
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 2)
    return max(1, min(12, cpus - 2))


_CPU_STATE = None


def cpu_init(H, L, pri):
    """Child processes hold the matrices; nothing here touches CUDA."""
    global _CPU_STATE
    for _p in (HERE, CIRCMEAN, ACC):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import exact_cascade
    _CPU_STATE = (H, L, pri, exact_cascade)


def cpu_task(chunk_id, det, obs, screen):
    H, L, pri, ec = _CPU_STATE
    (sf, po, un, st, ns), diag = ec.cascade(H, L, pri, det, obs, screen, mode=MODE)
    diag["chunk"] = int(chunk_id)
    diag["pid"] = os.getpid()
    diag["dec3_builds_in_process"] = ec.stage3_stats()["builds"]
    return (int(sf), np.asarray(po, np.int64), int(un), [int(x) for x in st],
            int(ns), diag)


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.isf(alpha / 2, k + 1, n - k))
    return lo, hi


def units(codes):
    """Every (basis, code) pair: bases outer, codes inner."""
    return [(b, c) for b in BASES for c in codes]


def assignment(nunits):
    """Map this Slurm task to a unit and to a slice of that unit's chunks.

    Rank r takes unit r % nunits; the ranks sharing a unit split its chunk ids
    round-robin. Outside Slurm, or with one task, this is unit-per-loop and one
    shard, which is exactly the serial driver.
    """
    rank = os.environ.get("SLURM_PROCID")
    world = os.environ.get("SLURM_NTASKS")
    # A plain sbatch script sets both even with one task; sharding one task is
    # the serial run with an extra part file, so only branch for a real srun.
    if rank is None or world is None or int(world) < 2:
        return None, 0, 1, 0, 1
    rank, world = int(rank), int(world)
    if not 0 <= rank < world:
        raise ValueError("SLURM_PROCID %d outside SLURM_NTASKS %d" % (rank, world))
    unit = rank % nunits
    shard_count = len(range(unit, world, nunits))
    shard_index = rank // nunits
    return unit, shard_index, shard_count, rank, world


def auto_group(H, piece):
    """Chunks per GPU call that fit, from the arrays decode() actually holds.

    decode keeps q and cv at one float per (edge, column), post and out_post at
    one float per (variable, column), hard/out_hard a byte each and syn a byte
    per (check, column); compaction allocates a second copy of q while the old
    one is alive. Ask for MEM_FRAC of what is free and round down to chunks.
    """
    import torch
    if GROUP:
        return GROUP
    edges, nvar, nchk = int(H.nnz), int(H.shape[1]), int(H.shape[0])
    per_shot = 4 * (3 * edges + (2 if DETAILS else 1) * nvar) + 2 * nvar + nchk
    free, _total = torch.cuda.mem_get_info(torch.cuda.current_device())
    shots = int(MEM_FRAC * free / max(1, per_shot))
    return int(max(1, min(GROUP_CAP, shots // max(1, piece))))


def slice_screen(screen, a, b, det_chunk):
    """One chunk's view of a grouped screen, with its own detector fingerprint.

    The fingerprint is recomputed for the slice; it is metadata binding a screen
    to its inputs, and the inputs of this slice are this chunk's detectors.
    """
    from screen_contract import detectors_fingerprint
    out = dict(protocol=screen["protocol"],
               success=screen["success"][a:b],
               iterations=screen["iterations"][a:b],
               unsafe_mask=screen["unsafe_mask"][a:b],
               detectors_fingerprint=detectors_fingerprint(det_chunk))
    for key in ("observables", "decoding", "posterior"):
        if key in screen:
            out[key] = screen[key][a:b]
    if "observables_fingerprint" in screen:
        out["observables_fingerprint"] = screen["observables_fingerprint"]
    return out


class Sampler(object):
    """Draws the seeded chunks, one group ahead of the decoder.

    Chunk c is always sampler(seed=1000 + CHUNK*c).sample(len) -- the thread
    changes when a chunk is drawn, never what is drawn.
    """

    def __init__(self, circ, groups, shared=None, attrib=None):
        self.circ, self.groups, self.i = circ, groups, 0
        self.shared = shared
        self.attrib = attrib          # pl_attrib.FrameAttrib or None
        self.next = None
        self.error = None
        self.thread = None
        self.lock = threading.Lock()

    def draw(self, group):
        out = []
        for (c, seed, m) in group:
            if self.attrib is not None:
                # frame simulator, so the final Pauli frame and the flipped
                # measurements of every shot are available for the failure map;
                # same statistics, but a different random stream than the
                # detector sampler with the same seed
                import stim
                sim = stim.FlipSimulator(batch_size=m, seed=int(seed),
                                         disable_stabilizer_randomization=True)
                sim.do(self.circ)
                det = np.ascontiguousarray(sim.get_detector_flips().T.astype(np.uint8))
                ob = np.ascontiguousarray(sim.get_observable_flips().T.astype(np.uint8))
                extra = self.attrib.per_shot(sim, m)
                out.append((c, det, ob, extra))
                continue
            sampler = (self.shared if seed is None
                       else self.circ.compile_detector_sampler(seed=seed))
            det, ob = sampler.sample(m, separate_observables=True)
            out.append((c, np.asarray(det, np.uint8), np.asarray(ob, np.uint8), None))
        return out

    def _prefetch(self, group):
        try:
            self.next = self.draw(group)
        except BaseException as exc:                       # re-raised on the main thread
            self.error = exc

    def get(self, i):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            if self.error is not None:
                raise self.error
            value, self.next = self.next, None
        else:
            value = self.draw(self.groups[i])
        if PREFETCH and i + 1 < len(self.groups):
            self.thread = threading.Thread(target=self._prefetch,
                                           args=(self.groups[i + 1],), daemon=True)
            self.thread.start()
        return value


def run_unit(code, basis, sched, decoder_cache, shard_index, shard_count, timers):
    import torch
    import pl_at_p_asc as ASC
    import pl_at_p as PL
    import circ_1side as C1
    from gpu_backend import make_first_leg
    import exact_cascade as EC
    from screen_contract import detectors_fingerprint
    if MODE == "gpu2" and not os.path.abspath(EC.__file__).startswith(PLSTAGE2):
        raise ImportError("PLC_MODE=gpu2 needs %s/exact_cascade.py, got %s"
                          % (PLSTAGE2, EC.__file__))

    t0 = time.time()
    ts = time.perf_counter()
    circ, n, k, depth = ASC.build(code["npz"], int(code["q"]), basis,
                                  code["name"], sched)
    dem = circ.detector_error_model(decompose_errors=False,
                                    approximate_disjoint_errors=True)
    H, obsm, pri = C1.dem_matrices(dem)
    key = (code["name"], basis)
    if key not in decoder_cache:
        dev = "cuda:%d" % (torch.cuda.current_device() if torch.cuda.is_available() else 0)
        decoder_cache[key] = make_first_leg(
            H, pri, gamma0=PL.STAGE1["gamma0"], device=dev, L=obsm,
            implementation=GPU_IMPL)
    decoder = decoder_cache[key]
    timers["setup"] = time.perf_counter() - ts

    shared = None
    if ONESAMPLER:
        if shard_count != 1:
            raise ValueError("PLF_ONESAMPLER draws one sequential stream; it "
                             "cannot be split across shards")
        piece = BATCH
        chunks = [(c, None, min(piece, SHOTS - c * piece))
                  for c in range(int(math.ceil(SHOTS / float(piece))))]
        mine = chunks
        shared = circ.compile_detector_sampler(seed=SEED0)
    else:
        piece = CHUNK
        chunks = [(c, 1000 + c * CHUNK, min(CHUNK, SHOTS - c * CHUNK))
                  for c in range(int(math.ceil(SHOTS / float(CHUNK))))]
        mine = [t for t in chunks if t[0] % shard_count == shard_index]
    g = auto_group(H, piece)
    groups = [mine[i:i + g] for i in range(0, len(mine), g)]
    timers["group"] = g
    timers["piece"] = piece

    state = dict(shot_fail=0, unsat=0, stages=[0, 0, 0], done=0,
                 per_obs=np.zeros(obsm.shape[0], dtype=np.int64), slowest=0.0)
    # block-level failure map from the scored shots (needs the shapes)
    attrib = None
    if ATTRIB and code.get("shapes"):
        if ONESAMPLER:
            raise ValueError("PLF_ATTRIB needs seeded chunks; PLF_ONESAMPLER=1 is not supported with it")
        import pl_attrib
        _z = np.load(code["npz"])
        _hx = np.asarray(_z["HX"], np.uint8) % 2
        _hz = np.asarray(_z["HZ"], np.uint8) % 2
        _h1, _h2 = (_hx, _hz) if basis == "Z" else (_hz, _hx)
        attrib = pl_attrib.FrameAttrib(circ, n, int(code["q"]), tuple(code["shapes"]), basis,
                                       _h1.shape[0], _h2.shape[0])
    stopped_on = "shots"
    if MAX_ERRORS and shard_count != 1:
        # Shards would each stop on their own error count, so the union is not
        # any single stopping rule and the shot counts would not add.
        raise ValueError("PL_MAX_ERRORS and sharding cannot be combined "
                         "(shard %d of %d)" % (shard_index, shard_count))
    # per-chunk counts, written to CHUNKLOG if set
    chunk_log = {}

    def accumulate(result, expect, chunk_id=None, extra=None):
        sf, po, un, st, ns, diag = result
        if attrib is not None and extra is not None:
            attrib.add(extra, diag.get("failed_idx"))
        if int(ns) != expect:
            raise AssertionError("continuation returned %d of %d shots" % (ns, expect))
        state["shot_fail"] += int(sf)
        state["per_obs"] += np.asarray(po, np.int64)
        state["unsat"] += int(un)
        for i in range(3):
            state["stages"][i] += int(st[i])
        state["done"] += expect
        cid = diag.get("chunk", chunk_id)
        po = np.asarray(po, np.int64)
        chunk_log[int(cid)] = dict(
            chunk=int(cid), shots=int(ns), shot_failures=int(sf), unsat=int(un),
            stages=[int(x) for x in st], logical_total=int(po.sum()),
            per_obs_nonzero={str(j): int(po[j]) for j in np.flatnonzero(po)},
            seconds=diag.get("seconds"), cascade_shots=diag.get("shots", []),
            dec3_builds_in_call=diag.get("dec3_builds_in_call", 0),
            dec3_builds_in_process=diag.get("dec3_builds_in_process", 0),
            pid=diag.get("pid"))

    nw = worker_count()
    pool = None
    if nw:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        ts = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"),
                                   initializer=cpu_init, initargs=(H, obsm, pri))
        timers["pool_spawn"] = time.perf_counter() - ts
    pending = {}

    def harvest(block):
        if not pending:
            return
        if block:
            from concurrent.futures import FIRST_COMPLETED, wait
            ready, _rest = wait(list(pending), return_when=FIRST_COMPLETED)
        else:
            ready = [f for f in list(pending) if f.done()]
        for f in ready:
            expect, started, cid, extra = pending.pop(f)
            accumulate(f.result(), expect, cid, extra)
            state["slowest"] = max(state["slowest"], time.perf_counter() - started)

    src = Sampler(circ, groups, shared, attrib)
    try:
        for gi in range(len(groups)):
            tA = time.perf_counter()
            drawn = src.get(gi)
            timers["sample"] = timers.get("sample", 0.0) + time.perf_counter() - tA
            det_cat = np.concatenate([d for (_c, d, _o, _e) in drawn], 0)
            before = detectors_fingerprint(det_cat)
            tA = time.perf_counter()
            screen = decoder.decode(det_cat, pre_iter=PL.STAGE1["pre_iter"],
                                    check_every=8, return_details=DETAILS)
            timers["gpu"] = timers.get("gpu", 0.0) + time.perf_counter() - tA
            # Each chunk's slice gets its own fingerprint below, which would be a
            # tautology on its own. This is the check that still bites: the screen
            # belongs to the detectors as they went in, unmodified.
            if screen.get("detectors_fingerprint") != before:
                raise AssertionError("the decoder changed the sampled detectors")
            tA = time.perf_counter()
            off = 0
            for (_c, det, ob, extra) in drawn:
                m = len(det)
                sub = slice_screen(screen, off, off + m, det)
                off += m
                if pool is not None and not np.asarray(sub["success"], bool).all():
                    # This chunk needs the relay cascade: hand it to a worker and
                    # keep the GPU moving. Counts are integers and add in any order.
                    fut = pool.submit(cpu_task, _c, det, ob, sub)
                    pending[fut] = (m, time.perf_counter(), _c, extra)
                    timers["chunks_offloaded"] = timers.get("chunks_offloaded", 0) + 1
                else:
                    # Fully screened chunk: no decoder is built and no relay RNG
                    # is drawn, so every mode does the same thing here.
                    res, diag = EC.cascade(H, obsm, pri, det, ob, sub, mode=MODE)
                    diag["chunk"] = int(_c)
                    diag["pid"] = os.getpid()
                    accumulate(tuple(res) + (diag,), m, _c, extra)
                if MAX_ERRORS:
                    harvest(block=False)
                    # the cap counts failed shots, not flipped logicals
                    if int(state["shot_fail"]) >= MAX_ERRORS:
                        stopped_on = "errors"
                        break
            timers["cpu_inline"] = timers.get("cpu_inline", 0.0) + time.perf_counter() - tA
            harvest(block=False)
            # The cap only bounds host memory (a queued task is one chunk's
            # detectors, about half a megabyte). Blocking here would idle the
            # GPU behind a slow chunk, which is the thing being avoided.
            while len(pending) >= max(64, 8 * max(1, nw)):
                harvest(block=True)
            if stopped_on == "errors":
                # Drain and count everything still in flight: dropping it would
                # discard the slowest (mostly failing) shots and bias pL downward.
                tB = time.perf_counter()
                while pending:
                    harvest(block=True)
                timers["cpu_drain"] = timers.get("cpu_drain", 0.0) + \
                    time.perf_counter() - tB
                break
            if PROGRESS and (gi + 1) % PROGRESS == 0:
                print("  [%s %s shard %d/%d] %d/%d shots settled, %d chunks in "
                      "flight, %.1f s" % (code["name"], basis, shard_index,
                                          shard_count, state["done"],
                                          sum(t[2] for t in mine), len(pending),
                                          time.time() - t0), flush=True)
        tA = time.perf_counter()
        while pending:
            harvest(block=True)
        timers["cpu_drain"] = timers.get("cpu_drain", 0.0) + \
            time.perf_counter() - tA
    finally:
        if src.thread is not None:
            src.thread.join()          # a prefetch left running by an early stop
        if pool is not None:
            pool.shutdown(wait=True)

    assigned = sum(t[2] for t in mine)
    if state["done"] > assigned or state["done"] == 0:
        raise AssertionError("settled %d shots, assigned %d"
                             % (state["done"], assigned))
    if stopped_on == "shots" and state["done"] != assigned:
        raise AssertionError("settled %d shots, assigned %d"
                             % (state["done"], assigned))
    timers["workers"] = nw
    timers["slowest_offloaded_chunk"] = round(state["slowest"], 1)
    timers["mode"] = MODE
    timers["stage3"] = EC.STAGE3
    timers["dec3_builds"] = sum(r["dec3_builds_in_call"] for r in chunk_log.values())
    if CHUNKLOG:
        tag = "%s_%s_%s" % (code["name"], basis, MODE)
        path = "%s.%s.json" % (CHUNKLOG, tag)
        json.dump([chunk_log[c] for c in sorted(chunk_log)], open(path, "w"), indent=1)
        print("    per-chunk record -> %s" % path, flush=True)
    timers["stopped_on"] = stopped_on
    return dict(name=code["name"], basis=basis, n=n, k=k, d=code.get("d"),
                shots=state["done"], depth=depth, stopped_on=stopped_on,
                shot_failures=state["shot_fail"],
                per_obs=[int(x) for x in state["per_obs"]],
                unsatisfied_syndromes=state["unsat"],
                resolved_by_stage=state["stages"], secs=round(time.time() - t0, 1),
                attrib=(attrib.summary() if attrib is not None else None),
                shard_index=shard_index, shard_count=shard_count,
                group_chunks=g, timers={x: (round(y, 2) if isinstance(y, float) else y)
                                        for x, y in timers.items()})


def finalise(raw):
    """Output record of one unit from its (summed) counts."""
    N, k = raw["shots"], raw["k"]
    per = np.asarray(raw["per_obs"], np.int64)
    shot_fail = raw["shot_failures"]
    lo, hi = clopper_pearson(shot_fail, N)
    tot = int(per.sum())
    lo_o, hi_o = clopper_pearson(tot, N * k)
    p_obs = tot / float(N * k)
    # per-observable convention (mean over logicals), kept for reference
    per_round_hi_obs = 1.0 - (1.0 - min(hi_o, 1.0)) ** (1.0 / ROUNDS)
    ceil_hi_obs = 1.0 - (1.0 - clopper_pearson(0, N * k)[1]) ** (1.0 / ROUNDS)
    # score convention: the shot failure rate shot_fail/N, per round, divided
    # by k; the interval is on shot_fail out of N independent shots, not on tot
    # out of N*k.
    p_shot = shot_fail / float(N)
    per_round_block = 1.0 - (1.0 - min(p_shot, 1.0)) ** (1.0 / ROUNDS)
    per_round_block_hi = 1.0 - (1.0 - min(hi, 1.0)) ** (1.0 / ROUNDS)
    per_round_hi = per_round_block_hi / k
    # What a zero-failure sample of this size would have scored: the ceiling the
    # score saturates at. It depends on N and k and so differs per code.
    ceil_hi = (1.0 - (1.0 - clopper_pearson(0, N)[1]) ** (1.0 / ROUNDS)) / k
    return dict(name=raw["name"], basis=raw["basis"], n=raw["n"], k=k, d=raw.get("d"),
                p=P, rounds=ROUNDS, shots=N, depth=raw["depth"],
                shot_failures=shot_fail, pl_shot=shot_fail / float(N),
                pl_shot_ci95=[lo, hi], logical_failures_total=tot,
                pl_shot_per_logical=p_obs, pl_shot_per_logical_ci95=[lo_o, hi_o],
                pl_shot_per_logical_max=float(per.max()) / N,
                worst_logical=int(per.argmax()),
                # per-observable convention
                pl_round_per_logical_approx=(1.0 - (1.0 - p_obs) ** (1.0 / ROUNDS))
                if p_obs < 1 else 1.0,
                pl_round_per_logical_ci95_hi=per_round_hi_obs,
                score_perobs=-math.log10(max(per_round_hi_obs, 1e-12)),
                score_ceiling_perobs=-math.log10(max(ceil_hi_obs, 1e-12)),
                # standard convention: block rate per round, per logical qubit
                pl_round_block=per_round_block,
                pl_round_block_ci95_hi=per_round_block_hi,
                pl_round_per_logical_std=per_round_block / k,
                pl_round_per_logical_std_ci95_hi=per_round_hi,
                score=-math.log10(max(per_round_hi, 1e-12)),
                score_ceiling=-math.log10(max(ceil_hi, 1e-12)),
                max_errors=MAX_ERRORS, max_shots=SHOTS,
                stopped_on=raw.get("stopped_on", "shots"),
                per_obs=[int(x) for x in per],
                logicals_failed=int((per > 0).sum()),
                unsatisfied_syndromes=raw["unsatisfied_syndromes"],
                resolved_by_stage=raw["resolved_by_stage"],
                decoder="GPU first leg (%s) + CPU cascade" % GPU_IMPL,
                circuit="asc_schedule interleaved + depolarising",
                secs=raw["secs"], shards=raw.get("shards", 1),
                attrib=raw.get("attrib"),
                group_chunks=raw.get("group_chunks"), timers=raw.get("timers"))


def merge(parts):
    """Add the per-shard integer counts of one unit."""
    head = dict(parts[0])
    head["per_obs"] = [int(x) for x in np.sum([p["per_obs"] for p in parts], axis=0)]
    head["shots"] = sum(p["shots"] for p in parts)
    head["shot_failures"] = sum(p["shot_failures"] for p in parts)
    head["unsatisfied_syndromes"] = sum(p["unsatisfied_syndromes"] for p in parts)
    head["resolved_by_stage"] = [sum(p["resolved_by_stage"][i] for p in parts)
                                 for i in range(3)]
    head["secs"] = max(p["secs"] for p in parts)
    head["shards"] = len(parts)
    if len(parts) > 1 and all(p.get("attrib") and p["attrib"].get("raw") for p in parts):
        import pl_attrib
        head["attrib"] = pl_attrib.FrameAttrib.merge_summaries([p["attrib"] for p in parts])
    elif len(parts) > 1 and any(p.get("attrib") for p in parts):
        head["attrib"] = None
        head["attrib_error"] = "shard attributions without raw counts cannot be merged"
    return head


def main(codes_path, out_path):
    sched = json.load(open(os.environ.get(
        "ASC_SCHED", "/u/zdliu/lp/flagship_pl_20260910/asc_schedules.json")))
    codes = json.load(open(codes_path))
    todo = units(codes)
    unit, shard_index, shard_count, rank, world = assignment(len(todo))
    cache = {}
    if unit is None:
        out = []
        for basis, code in todo:
            raw = run_unit(code, basis, sched, cache, 0, 1, {})
            rec = finalise(raw)
            out.append(rec)
            t = raw["timers"]
            print("%-26s %s [[%d,%d]] depth %d shots %d (stopped on %s; caps "
                  "%d errors / %d shots): fail %d, logical %d, "
                  "stages %s, unsat %d, worst %d, score %.3f of %.3f, %.0f s"
                  % (rec["name"], rec["basis"], rec["n"], rec["k"], rec["depth"],
                     rec["shots"], rec["stopped_on"], rec["max_errors"],
                     rec["max_shots"], rec["shot_failures"],
                     rec["logical_failures_total"], rec["resolved_by_stage"],
                     rec["unsatisfied_syndromes"], rec["worst_logical"],
                     rec["score"], rec["score_ceiling"], rec["secs"]),
                  flush=True)
            print("    mode %s/%s: group %s chunks/call, %s workers, setup %.0f s,"
                  " sample %.0f s, gpu %.0f s, cpu inline %.0f s, drain %.0f s,"
                  " slowest chunk %.0f s, %s chunks offloaded, %s stage-3 builds"
                  % (t.get("mode"), t.get("stage3"), t.get("group"),
                     t.get("workers"), t.get("setup", 0),
                     t.get("sample", 0), t.get("gpu", 0), t.get("cpu_inline", 0),
                     t.get("cpu_drain", 0), t.get("slowest_offloaded_chunk", 0),
                     t.get("chunks_offloaded", 0), t.get("dec3_builds")), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
        print("wrote %s" % out_path, flush=True)
        return

    basis, code = todo[unit]
    raw = run_unit(code, basis, sched, cache, shard_index, shard_count, {})
    raw["unit"] = unit
    part = "%s.part%04d.json" % (out_path, rank)
    json.dump(raw, open(part, "w"))
    print("rank %d/%d -> unit %d (%s %s) shard %d/%d: %d shots, %.0f s -> %s"
          % (rank, world, unit, code["name"], basis, shard_index, shard_count,
             raw["shots"], raw["secs"], part), flush=True)


def collect(out_path, codes_path):
    """Sum the .partNNNN.json files of a sharded run into one output file."""
    import glob
    sched_codes = json.load(open(codes_path))
    order = {(b, c["name"]): i for i, (b, c) in enumerate(units(sched_codes))}
    by_unit = {}
    for path in sorted(glob.glob(out_path + ".part*.json")):
        raw = json.load(open(path))
        by_unit.setdefault(raw["unit"], []).append(raw)
    out = []
    expected = set(range(len(order)))
    if set(by_unit) != expected:
        raise AssertionError("units present %s, expected %s"
                             % (sorted(by_unit), sorted(expected)))
    for u in sorted(by_unit):
        parts = by_unit[u]
        got = parts[0]["shard_count"]
        idx = sorted(p["shard_index"] for p in parts)
        if idx != list(range(got)) or any(p["shard_count"] != got for p in parts):
            raise AssertionError("unit %d: shard indices %s of %d shards (duplicate or missing part files)"
                                 % (u, idx, got))
        for key in ("name", "basis", "n", "k", "depth"):
            if len({json.dumps(p.get(key)) for p in parts}) != 1:
                raise AssertionError("unit %d: shards disagree on %s" % (u, key))
        rec = finalise(merge(parts))
        if rec["shots"] != SHOTS and rec.get("stopped_on") != "errors":
            raise AssertionError("unit %d: %d shots, expected %d"
                                 % (u, rec["shots"], SHOTS))
        out.append(rec)
        print("%-26s %s [[%d,%d]] shots %d: fail %d, logical %d, stages %s, "
              "unsat %d, worst %d, %.0f s wall over %d shards"
              % (rec["name"], rec["basis"], rec["n"], rec["k"], rec["shots"],
                 rec["shot_failures"], rec["logical_failures_total"],
                 rec["resolved_by_stage"], rec["unsatisfied_syndromes"],
                 rec["worst_logical"], rec["secs"], rec["shards"]), flush=True)
    out.sort(key=lambda r: order[(r["basis"], r["name"])])
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote %s" % out_path, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
