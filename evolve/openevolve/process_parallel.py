"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import (
    Program,
    ProgramDatabase,
    bin_islands_enabled,
    build_lineage_chain,
)
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None


def _worker_init(config_dict: dict, evaluation_file: str) -> None:
    """Initialize worker process with necessary components"""
    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    from openevolve.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        LLMConfig,
        PromptConfig,
        LLMModelConfig,
    )

    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    # Create LLM config with models
    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    # Create other configs
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["llm", "prompt", "database", "evaluator"]
        },
    )
    _worker_evaluation_file = evaluation_file

    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _round_level(db_snapshot, parent):
    """The assignment level this round runs at, or None to deal from the deck.

    A quota round overrides whatever the parent asked for; that is what stops a
    model from staying in L1 and what keeps all three levels producing samples.
    Otherwise the parent's own declaration decides, and a parent that declared
    nothing falls through to the deck.
    """
    from openevolve.prompt.assignment import parse_next_move, self_selection_enabled

    if not self_selection_enabled():
        return None
    forced = db_snapshot.get("forced_level")
    if forced:
        return int(forced)
    level, _why = parse_next_move(getattr(parent, "code", "") or "")
    return level


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler

        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,  # No shared database in worker
        )


def _cm_bin_inspirations(programs, parent, k, seed):
    '''circmean: the inspirations are the elites of the
    OTHER rate bins (metrics rate_bin), drawn uniformly, from the WHOLE
    population - not the parent's island. Returns [] when k == 0.'''
    import random as _random
    pb = parent.metrics.get("rate_bin")
    elites = {}
    for p in programs.values():
        b = p.metrics.get("rate_bin")
        if b is None or b == pb or p.id == parent.id:
            continue
        sc = p.metrics.get("combined_score", 0) or 0
        if b not in elites or sc > (elites[b].metrics.get("combined_score", 0) or 0):
            elites[b] = p
    pool = sorted(elites.values(), key=lambda p: p.metrics.get("rate_bin"))
    rng = _random.Random(seed)
    return rng.sample(pool, min(k, len(pool)))


def _cell_focus_levels():
    from openevolve.prompt.sampler import cell_focus_levels
    return cell_focus_levels()


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]
        _cm_k = int(os.environ.get("CM_INSP_BINS", "0") or 0)
        if _cm_k > 0:
            inspirations = _cm_bin_inspirations(programs, parent, _cm_k, iteration)

        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

        # Get island-specific programs for context
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs
        ]

        # Sort by metrics for top programs
        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True,
        )

        # Use config values for limits instead of hardcoding
        # TOP PROGRAMS: one elite per occupied k_bin (grid representatives), so the
        # LLM sees the whole k-vs-depth map, not N deepest of one lineage. Falls back
        # to island-fitness order if no k_bin metric present.
        _nshow = _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        _bybin = {}
        # CM_TOP_GLOBAL=1: the top section shows the best programs of the
        # PARENT'S OWN rate band from the whole population (all islands);
        # the other bands are covered by the inspirations (CM_INSP_BINS)
        _top_pool = island_programs
        if os.environ.get("CM_TOP_GLOBAL", "0") == "1":
            _pbin = parent.metrics.get("rate_bin")
            _top_pool = sorted(
                [_p for _p in programs.values() if _p.metrics.get("rate_bin") == _pbin],
                key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
                reverse=True)
            _bybin = None
        for _p in (_top_pool if _bybin is not None else []):
            _kb = _p.metrics.get("k_bin")
            if _kb is None: continue
            _kb = int(_kb); _sc = _p.metrics.get("combined_score", safe_numeric_average(_p.metrics))
            if _kb not in _bybin or _sc > _bybin[_kb].metrics.get("combined_score", safe_numeric_average(_bybin[_kb].metrics)):
                _bybin[_kb] = _p
        if _bybin:
            _reps = sorted(_bybin.values(), key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)), reverse=True)
            _cands = _reps
        else:
            _cands = _top_pool
        # 2026-09-17: under CM_TOP_GLOBAL=1 the pool is the parent's own rate
        # bin. In a narrow cell every candidate carries the parent's own code,
        # the code-deduplication below then removes all of them and the section
        # is empty (66 of 162 audited rounds). Widen the pool in two steps -
        # neighbouring bins, then the island - so the section always shows
        # something, ordered by score.
        if os.environ.get("CM_TOP_GLOBAL", "0") == "1":
            _pb = parent.metrics.get("rate_bin")
            def _score(_p):
                return _p.metrics.get("combined_score", safe_numeric_average(_p.metrics))
            def _distinct(_lst):
                _out, _codes = [], {parent.code}
                for _p in _lst:
                    if _p.id == parent.id or _p.code in _codes:
                        continue
                    _codes.add(_p.code)
                    _out.append(_p)
                return _out
            if len(_distinct(_cands)) < _nshow and _pb is not None:
                _near = [_p for _p in programs.values()
                         if _p.metrics.get("rate_bin") is not None
                         and abs(float(_p.metrics["rate_bin"]) - float(_pb)) <= 1]
                _cands = _cands + sorted(_near, key=_score, reverse=True)
            if len(_distinct(_cands)) < _nshow:
                _cands = _cands + sorted(island_programs, key=_score, reverse=True)
        # 2026-09-09: never show the parent again (it is the Current Program)
        # and never show two programs with identical code
        programs_for_prompt = []
        _seen_code = {parent.code}
        for _p in _cands:
            if _p.id == parent.id or _p.code in _seen_code:
                continue
            _seen_code.add(_p.code)
            programs_for_prompt.append(_p)
            if len(programs_for_prompt) >= _nshow:
                break
        # PREVIOUS ATTEMPTS: the parent's OWN lineage, walked back through
        # parent_id, chronological (oldest first, the parent's direct parent
        # last). It used to be island_programs[:N] - the island leaderboard -
        # so the section showed unrelated programs and the fitness delta in
        # sampler.py compared the parent against the island's third-best
        # program. Ancestors are looked up in the full snapshot, not just the
        # island slice, because migration can leave an ancestor on another
        # island. An empty chain (seed program, or ancestor pruned from the
        # snapshot) is passed through as empty: the sampler then omits the
        # delta line rather than comparing against something unrelated.
        lineage_programs = build_lineage_chain(
            programs, parent.id, _worker_config.prompt.num_previous_programs
        )

        import os as _os
        # v1: inject each shown program's own reasoning as a code comment
        if _os.environ.get('REASONING_IN_PROMPT') == '1':
            def _inj(p):
                r = (p.metadata or {}).get('reasoning', '') if hasattr(p, 'metadata') else ''
                if r:
                    p = p  # Program is mutable dict-backed; append to a copy of code
                return r
            _rmap = {}
            for _p in list(programs_for_prompt) + list(inspirations) + list(lineage_programs) + [parent]:
                _r = (_p.metadata or {}).get('reasoning', '')
                if _r: _rmap[_p.id] = _r
        else:
            _rmap = {}
        def _with_reasoning(pdict):
            rid = pdict.get('id'); r = _rmap.get(rid, '')
            if r:
                pdict = dict(pdict)
                pdict['code'] = pdict.get('code', '') + '\n\n# --- DESIGN REASONING (this program) ---\n# ' + r.replace('\n', ' ')[:int(os.environ.get("OE_REASONING_CODE_CHARS", "700"))]
            return pdict
        # Build prompt
        _v6_stay = False
        try:
            _pm = getattr(parent, "metadata", None) or {}
            if (_pm.get("fresh_bed") and int(_pm.get("fresh_children", 0) or 0) < 2
                    and int(iteration) - int(_pm.get("fresh_iter", iteration) or iteration) <= 20):
                _v6_stay = True
        except Exception:
            _v6_stay = False
        _v6_level = 1 if _v6_stay else _round_level(db_snapshot, parent)
        prompt = _worker_prompt_sampler.build_prompt(
            current_program=(parent.code + ('\n\n# --- DESIGN REASONING (this program) ---\n# ' + _rmap[parent.id].replace('\n',' ')[:int(os.environ.get("OE_REASONING_CODE_CHARS", "700"))] if parent.id in _rmap else '')),
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[_with_reasoning(p.to_dict()) for p in lineage_programs],
            top_programs=[_with_reasoning(p.to_dict()) for p in programs_for_prompt],
            inspirations=[_with_reasoning(p.to_dict()) for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
            # Lets the sampler verify that previous_programs[-1] really is this
            # program's parent before it prints a fitness delta.
            current_program_parent_id=parent.parent_id,
            # The corner this round serves (the max row weight of the cell the
            # parent was drawn from, None when it came from no cell) and the
            # run's best result at every weight. Both are computed in the
            # controller, from the archive - a worker only sees a snapshot.
            cell_target=db_snapshot.get("cell_target"),
            cell_target_rate=db_snapshot.get("cell_target_rate"),
            weight_frontier=db_snapshot.get("weight_frontier"),
            # Reseeds the sampler for this round only. Without it the card and
            # the template variation are whatever this process drew first, and
            # a fresh process per generation makes that the same value all run.
            round_seed=db_snapshot.get("prompt_round_seed"),
            # V2_CARD_SELF: the parent decides its children's level unless this
            # round is a quota round. Off (the default) both are None and the
            # card is dealt from the deck exactly as before.
            # v6 stay rule: a parent on a fresh bed gets two level-1 children first
            assignment_level=_v6_level,
        )

        # The GRID STATUS block was removed here, not repaired. It appended the
        # archive's occupancy map to the END of the user message, told the
        # model that "holding a cell is what buys reproduction", and listed the
        # empty cells - i.e. it handed over the selection mechanics and invited
        # the model to optimise its own reproduction probability instead of the
        # objective (PROMPT_DESIGN section 7). It was also wrong on its own
        # terms: it enumerated cells the binner cannot produce, and it labelled
        # every occupied bin with the POLICY winner's [[n,k]] rather than the
        # instance holding that bin, so two different bins printed an identical
        # [[1053,186]] - impossible when the bin is a function of k/n.
        # Appending anything here also silently defeated the assignment-first
        # ordering: the last thing the model read was the archive, not the
        # task.

        iteration_start = time.time()

        # Generate code modification (sync wrapper for async)
        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

        # Check for None response
        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary

            import os as _osr
            _child_reasoning = ''
            if _osr.environ.get('REASONING_IN_PROMPT') == '1':
                try:
                    from openevolve.llm.openai import LAST_REASONING_SUMMARY as _LRS
                    _child_reasoning = _LRS.get('text', '')
                    logger.info('REASONING_CAPTURED len=%d head=%s' % (len(_child_reasoning), _child_reasoning[:60]))
                except Exception as _e: logger.info('REASONING_CAPTURE_FAIL %s' % _e)
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error=f"No valid diffs found in response", iteration=iteration
                )

            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            _child_reasoning = locals().get('_child_reasoning', '')
            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error=f"No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # Evaluate the child program
        import uuid

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        # v6: the declared target cell ("TARGET CELL: [a, b)") and the dealt card level
        _v6_target = None
        try:
            import re as _re
            _m = _re.search(r"TARGET\s+CELL\s*:\s*\[?\s*([0-9.]+)\s*,\s*([0-9.]+)", llm_response or "", _re.I)
            if _m:
                _v6_target = [float(_m.group(1)), float(_m.group(2))]
            else:
                # v6b wording: a single cell CENTRE ("TARGET CELL: 0.225")
                _m = _re.search(r"TARGET\s+CELL\s*:\s*\[?\s*([0-9]*\.[0-9]+)", llm_response or "", _re.I)
                if _m:
                    _c = float(_m.group(1))
                    _v6_target = [round(_c - 0.0125, 4), round(_c + 0.0125, 4)]
        except Exception:
            _v6_target = None
        _v6_card_level = getattr(_worker_prompt_sampler, "last_card_level", None)
        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
                "reasoning": _child_reasoning,
                "card_level": _v6_card_level,
                "target_cell": _v6_target,
                # The cell the ROUND asked for (engine), distinct from
                # target_cell above, which is the band the MODEL declared for
                # itself. Kept so "asked for cell t, landed in cell l" is
                # measurable: the constraint is soft and is expected to miss.
                # recorded only when the round actually NAMED the cell to the model
                # (the block is part of the L2 card - see sampler.cell_focus_levels);
                # a round that asked for nothing cannot miss
                "round_cell": ((db_snapshot.get("cell_target_rate") or {}).get("cell")
                               if isinstance(db_snapshot.get("cell_target_rate"), dict)
                               and _v6_card_level in _cell_focus_levels()
                               else None),
                "forced_stay": bool(_v6_stay),
            },
        )

        iteration_time = time.time() - iteration_start

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        next_prompt_round_seed: Optional[Callable[[], int]] = None,
        forced_assignment_level: Optional[Callable[[], Optional[int]]] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        # Supplied by OpenEvolve, which outlives this object: a pool is created
        # and destroyed inside every run() call, so a counter kept here would
        # restart each generation. Left None (direct construction, tests) the
        # rounds are independently random rather than identical.
        self.next_prompt_round_seed = next_prompt_round_seed
        self.forced_assignment_level = forced_assignment_level

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations

        # Worker-to-island pinning for true island isolation
        self.num_islands = config.database.num_islands
        self.worker_island_map = {}

        # Distribute workers across islands using modulo
        for worker_id in range(self.num_workers):
            island_id = worker_id % self.num_islands
            self.worker_island_map[worker_id] = island_id

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
        logger.info(f"Worker-to-island mapping: {self.worker_island_map}")

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "language": config.language,
        }

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file),
        )

        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            # WEIGHT_ELITES: the cell the parent was just drawn from, as a max
            # row weight, and the run's best result at each weight. The snapshot
            # is taken AFTER sample(), so cell_target is this round's cell.
            "cell_target": self.database.cell_target_weight(),
            # The same question on the rate grid: which band this round is asked
            # to fill, and which band its parent came from. None on any other
            # grid and on a round that served no cell. getattr, because the two
            # files are backed up separately: restoring database.py alone would
            # otherwise make every iteration raise, return None, and the run
            # finish immediately having produced nothing.
            "cell_target_rate": getattr(
                self.database, "cell_target_rate", lambda: None)(),
            "weight_frontier": self.database.weight_frontier(),
            # This round's share of the run's randomness. It travels with the
            # round's other decisions because a worker cannot generate it: its
            # PromptSampler is a process-level global whose generator is built
            # once per process, and with one island a process handles exactly
            # one round before the pool is torn down.
            "prompt_round_seed": (
                self.next_prompt_round_seed() if self.next_prompt_round_seed else None
            ),
            # V2_CARD_SELF: which level this round is FORCED to, ignoring what
            # the parent asked for. Decided here because the round counter
            # lives on the controller; None means the round is free and the
            # parent's own NEXT_MOVE decides it.
            "forced_level": (
                self.forced_assignment_level() if self.forced_assignment_level else None
            ),
            "artifacts": {},  # Will be populated selectively
        }

        # Include artifacts for programs that might be selected
        # IMPORTANT: This limits artifacts (execution outputs/errors) to first 100 programs only.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The limit of 100 keeps
        # artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts

        return snapshot

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures by island to maintain distribution
        pending_futures: Dict[int, Future] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch - distribute across islands
        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        # Round-robin distribution across islands
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            logger.info(
                f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.convergence_threshold}, "
                f"metric={self.config.early_stopping_metric}"
            )
        else:
            logger.info("Early stopping disabled")

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                # Use evaluator timeout + buffer to gracefully handle stuck processes
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database (will auto-inherit parent's island)
                    # No need to specify target_island - database will handle parent island inheritance
                    self.database.add(child_program, iteration=completed_iteration)
                    # v6 fill-the-grid bookkeeping (stay rule): see patch_v6_fillgrid.py
                    try:
                        _v6_parent = self.database.programs.get(result.parent_id)
                        _cm = child_program.metadata or {}
                        if _v6_parent is not None and _cm.get("forced_stay"):
                            _v6_parent.metadata["fresh_children"] = int(_v6_parent.metadata.get("fresh_children", 0) or 0) + 1
                        _cl = _cm.get("card_level")
                        # v6c (design semantics): fresh_bed = an L2/L3 child that is now the
                        # elite of its cell; it is then queued as the forced parent of the
                        # next two iterations (see _submit_iteration)
                        _is_elite = child_program.id in set((self.database.feature_map or {}).values())
                        _cb = (child_program.metrics or {}).get("rate_bin")
                        _pb = (_v6_parent.metrics or {}).get("rate_bin") if _v6_parent is not None else None
                        if _cl in (2, 3) and _is_elite and _cb is not None and _cb != _pb and (child_program.metrics or {}).get("combined_score", 0):
                            child_program.metadata["fresh_bed"] = True
                            child_program.metadata["fresh_children"] = 0
                            child_program.metadata["fresh_iter"] = int(completed_iteration)
                            if not hasattr(self, "_v6_queue"):
                                self._v6_queue = []
                            if len(self._v6_queue) < 6:
                                self._v6_queue.append([child_program.id, 2])
                    except Exception as _e:
                        logger.warning(f"v6 bookkeeping skipped: {_e}")

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
                        and current_island_counter >= programs_per_island
                    ):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values() if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                break

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        # Get the metric to track for early stopping
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            # Default metric not found, use safe average (standard pattern)
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            # User specified a custom metric that doesn't exist
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
                            # Check for improvement
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                                logger.debug(
                                    f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                )
                            else:
                                iterations_without_improvement += 1
                                logger.debug(
                                    f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                )

                            # Check if we should stop
                            if (
                                iterations_without_improvement
                                >= self.config.early_stopping_patience
                            ):
                                self.early_stopping_triggered = True
                                logger.info(
                                    f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                    f"No improvement for {iterations_without_improvement} iterations "
                                    f"(best score: {best_score:.4f})"
                                )
                                break

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and continuing with next iteration."
                )
                # Cancel the future to clean up the process
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit next iterations maintaining island balance
            for island_id in range(self.num_islands):
                if (
                    len(island_pending[island_id]) < batch_per_island
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                ):
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
                        next_iteration += 1
                        break  # Only submit one iteration per completion to maintain balance

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(
        self, iteration: int, island_id: Optional[int] = None
    ) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            # Use specified island or current island
            target_island = island_id if island_id is not None else self.database.current_island

            # Bin-islands: this slot draws its parent from an OCCUPIED cell that is
            # allowed to reproduce, round-robin over that set. Empty cells are skipped
            # -- they fill by spillover, which is how the previous run grew 4 bins into
            # 15 -- and with fewer reproducing cells than slots the extra slots deepen
            # them instead of idling.
            #
            # WHICH cells may reproduce is asked of the database
            # (occupied_reproducing_cells -> reproducing_cell_keys), never decided here.
            # This code used to inline the RATE rule: read LAYOUT_RATE_BIN_N and drop
            # cell 0 and cell N-1 as out-of-window boundary bins. That rule is a fact
            # about rate_bin. On the weight grid the same two keys are w=6 and w=14 --
            # legitimate corners, and w=6 is precisely the light region the dimension
            # exists to develop -- so inlining it here sterilised them while the
            # database said they could reproduce, with nothing in the output saying so.
            if bin_islands_enabled():
                if self.database._is_rate_grid():
                    # Uniform over every cell allowed to reproduce, occupied or
                    # not: the round-robin below draws from
                    # occupied_reproducing_cells(), which can only deepen cells
                    # that already hold a program, so an empty band was never
                    # targeted. bin_target restricts the DRAW and is overwritten
                    # with the cell actually drawn, so the round's own target is
                    # kept separately for the prompt and the child's record.
                    _tc = self.database.pick_target_cell(iteration)
                else:
                    _occ = self.database.occupied_reproducing_cells()
                    _tc = _occ[target_island % len(_occ)] if _occ else None
                self.database.cell_target_key = _tc
                self.database.bin_target = _tc
            else:
                # Without bin-islands no round serves a cell, and the
                # exploitation sampler writes bin_target regardless - so the
                # round's target has to be cleared explicitly or the prompt
                # would name the cell the parent already sits in.
                self.database.cell_target_key = None

            # Temporarily set database to target island for sampling
            original_island = self.database.current_island
            self.database.current_island = target_island

            try:
                # Sample parent and inspirations from the target island
                parent, inspirations = self.database.sample(
                    num_inspirations=self.config.prompt.num_top_programs
                )
                # v6c: a fresh-bed elite gets its two follow-up rounds guaranteed
                _q = getattr(self, "_v6_queue", None) or []
                while _q:
                    _pid, _left = _q[0]
                    _fp = self.database.programs.get(_pid)
                    if _fp is None or int((_fp.metadata or {}).get("fresh_children", 0) or 0) >= 2:
                        _q.pop(0)
                        continue
                    parent = _fp
                    _q[0][1] = _left - 1
                    if _q[0][1] <= 0:
                        _q.pop(0)
                    logger.info(f"v6c stay: forced parent {_pid[:8]} (fresh bed), {_left - 1} follow-up(s) left")
                    break
                # The cell the parent REALLY came from, recorded after every
                # path that can change it: sample() and then the v6c stay queue
                # above, which replaces the parent without touching bin_target.
                # The prompt states this as a fact, so it is read off the parent
                # rather than off whichever cell key the draw happened to use.
                self.database.parent_cell_key = (
                    (getattr(parent, "metrics", None) or {}).get("rate_bin"))
            finally:
                # Always restore original island state
                self.database.current_island = original_island

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
