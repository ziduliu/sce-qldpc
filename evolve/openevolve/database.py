"""
Program database for OpenEvolve
"""

import base64
import json
import logging
import os
import math
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

# FileLock removed - no longer needed with threaded parallel processing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openevolve.config import DatabaseConfig
from openevolve.utils.code_utils import calculate_edit_distance
from openevolve.utils.metrics_utils import safe_numeric_average, get_fitness_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MAP-Elites over row weight (WEIGHT_ELITES.md)
#
# The evaluator emits, per policy, one float per max-row-weight it produced an
# instance at:  w_best_6 ... w_best_14, absent where the batch has no instance.
# A policy is registered in EVERY weight cell for which it holds the best
# instance seen so far, and every comparison about a cell uses that cell's
# number - never combined_score, which is the MAX over the policy's occupied
# cells and would otherwise let a heavy winner take every light cell it touches.
#
# The cell KEY is the archive coordinate, which is not the weight: WEIGHT_ELITES
# fixes bin = clamp(w, 6, 14) - 6, i.e. nine cells 0..8. The two are converted
# through _cell_weight / _weight_cell_key and nowhere else, because mixing them
# would silently alias cell "6" (weight 12) with weight 6.
# ---------------------------------------------------------------------------
WEIGHT_DIM = "weight_bin"
WEIGHT_MIN = 6
WEIGHT_MAX = 14
WEIGHT_METRIC_PREFIX = "w_best_"

# What the cell IS, next to what it scored: the evaluator publishes the holding
# candidate's index, its [[n,k,d]] and whether the deep distance reading agreed
# with the screen one, under wcell_<w>_<field> (evaluator_layout_bp_osd.py,
# W_CELL_PREFIX). Each field lists the spellings accepted, most-preferred first;
# a field the evaluator does not emit is left out of the line rather than
# guessed at.
WEIGHT_DETAIL_PREFIXES = ("wcell_", "w_best_")
_WEIGHT_DETAIL_KEYS = {
    "n": ("n",),
    "k": ("k",),
    "d": ("d", "d_fit"),
    "cand": ("idx", "cand", "c"),
    "converged": ("converged", "conv"),
}


# ---------------------------------------------------------------------------
# Bin-islands
#
# Under LAYOUT_BIN_ISLANDS=1 a program belongs to the island of its own cell and
# each slot draws its parent from one cell. WHICH cells may parent a child is a
# property of the coordinate, not a constant:
#
#   rate_bin     cell 0 and cell N-1 are the out-of-window boundary bins the
#                rate scheme deliberately builds (everything below the floor
#                lands in one, everything above the ceiling in the other), so
#                they hold programs that must not reproduce.
#   weight_bin   no such bins exist. All nine cells are legitimate corners, and
#                w=6 and w=14 are the extremes the axis was added to develop.
#
# A dimension not listed here has no known answer. Guessing either way changes
# what reproduces, silently, so bin-islands refuses to run on it instead.
# ---------------------------------------------------------------------------
BIN_ISLAND_DIMENSIONS = ("rate_bin", WEIGHT_DIM)


def bin_islands_enabled() -> bool:
    """True when this run draws parents per cell (LAYOUT_BIN_ISLANDS=1)."""
    return os.environ.get("LAYOUT_BIN_ISLANDS", "0") == "1"


# ---------------------------------------------------------------------------
# How wide the grid is
#
# Upstream OpenEvolve widens the configured bin count to
# ceil(archive_size ** (1/ndim)) so that "the grid can hold the archive". That
# is a reasonable default for a CONTINUOUS coordinate: the value is scaled into
# [0,1) and multiplied by the width, so the width is a resolution choice and a
# wider grid is simply a finer one.
#
# It is wrong for a PRE-BINNED coordinate. A *_bin metric IS the bin index,
# computed by the evaluator against fixed edges (_calculate_feature_coords), so
# the number of cells is a FACT about the coordinate rather than a knob:
# weight_bin has exactly nine legal values, bin = clamp(w, 6, 14) - 6. With
# archive_size 20 the widening turned the configured 9 into 20 and invented
# cells 9..19 that no program can ever occupy. Competition was unaffected (the
# keys really are 0..8) but every number derived from the width - the coverage
# denominator, the milestone log, neighbour/perturbation sampling, the width
# recorded in the checkpoint - was computed against a grid 2.2x the real one.
#
# So: a configured width for a discrete dimension is honoured verbatim, a
# per-dimension dict is honoured verbatim for every dimension it names, and only
# continuous dimensions are ever widened.
# ---------------------------------------------------------------------------

#: Width used for a dimension no configuration names (upstream default).
DEFAULT_FEATURE_BINS = 10


def is_discrete_dimension(dim: str) -> bool:
    """True when `dim` is a pre-binned coordinate with a fixed number of cells.

    The *_bin convention is the same one _calculate_feature_coords already
    follows ("a *_bin metric IS the bin index - no adaptive scaling"), so the
    two agree by construction. Everything else (score, complexity, diversity, a
    raw metric) is a continuous quantity scaled into the grid.
    """
    return str(dim).endswith("_bin")


def resolve_feature_bins(config: DatabaseConfig) -> Tuple[int, Dict[str, int]]:
    """(scalar default, per-dimension width) for the configured grid.

    The per-dimension map is authoritative; the scalar is only the fallback for
    a dimension the map does not name, and is what gets recorded in a
    checkpoint for a uniform grid.
    """
    dims = list(getattr(config, "feature_dimensions", None) or [])
    configured = getattr(config, "feature_bins", DEFAULT_FEATURE_BINS)

    if isinstance(configured, dict):
        # A per-dimension dict is an explicit statement about every dimension it
        # names. Nothing here may expand it, whatever the dimension is.
        return DEFAULT_FEATURE_BINS, {str(k): int(v) for k, v in configured.items()}

    if not dims:
        return 1, {}  # no MAP-Elites grid when feature_dimensions is empty

    width = int(configured)
    continuous = [dim for dim in dims if not is_discrete_dimension(dim)]
    widened = width
    if continuous:
        # Only the continuous axes may grow to hold the archive; the exponent
        # stays len(dims) because the archive has to fit in the whole product.
        archive_size = getattr(config, "archive_size", 0) or 0
        widened = max(width, int(pow(archive_size, 1 / len(dims)) + 0.99))
    per_dim = {dim: (width if is_discrete_dimension(dim) else widened) for dim in dims}
    return (widened if continuous else width), per_dim


def _finite_float(value: Any) -> Optional[float]:
    """value as a float, or None when it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if fv != fv or fv in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return fv


def weight_score(metrics: Optional[Dict[str, Any]], weight: int) -> Optional[float]:
    """The policy's deep-verified score at max row weight `weight`, or None.

    Absent and 0.0 stay distinguishable: absent means the batch built nothing at
    that weight, 0.0 means something was deep-verified there and scored nothing.
    """
    return _finite_float((metrics or {}).get("%s%d" % (WEIGHT_METRIC_PREFIX, weight)))


def weight_detail(metrics: Optional[Dict[str, Any]], weight: int) -> Dict[str, float]:
    """What the cell at `weight` IS: candidate index, n, k, d, converged.

    Fields the evaluator does not emit are absent from the result; callers omit
    them rather than substituting a number nothing measured.
    """
    metrics = metrics or {}
    out: Dict[str, float] = {}
    for field_name, suffixes in _WEIGHT_DETAIL_KEYS.items():
        for prefix in WEIGHT_DETAIL_PREFIXES:
            for suffix in suffixes:
                value = _finite_float(metrics.get("%s%d_%s" % (prefix, weight, suffix)))
                if value is not None:
                    out[field_name] = value
                    break
            if field_name in out:
                break
    return out


def _fmt_cell_score(value: Optional[float]) -> str:
    """A cell score for a log line. None is INELIGIBLE, not a number."""
    return "ineligible" if value is None else "%.3f" % value


# ---------------------------------------------------------------------------
# Archive schema contract (resume safety)
#
# feature_dimensions + feature_bins is not a contract: two runs can both say
# "weight_bin, 9 cells" and mean w=6..14 vs w=5..13, a cell score measured at
# 8 s vs 16 s, or k*d^2/n vs k*d^2/n^2. The stored grid is only reusable when
# the fields below agree, and load() names the one that does not.
# ---------------------------------------------------------------------------
ARCHIVE_SCHEMA_VERSION = 1

#: What a stored cell KEY means. A difference makes every key a different
#: coordinate, so the grid cannot be reused at all.
_SCHEMA_COORDINATE_FIELDS = (
    "feature_dimensions",
    "feature_definition",
    "feature_range",
    "feature_mapping",
    "cell_metric_prefix",
    # Only written when every active dimension is PRE-BINNED, where the number
    # of cells is a fact about the coordinate and a different one means the
    # keys were produced by a different binner (rate_bin at 17 bins vs 22 puts
    # a different rate in key "8"). It is deliberately not "feature_bins",
    # which on a continuous axis is a resolution choice that legitimately moves
    # with archive_size, and which older checkpoints recorded post-widening.
    "cell_count",
    # WHICH cells may parent a child. Part of what the archive IS, not a
    # setting: a grid grown with ten live cells cannot be continued as an
    # eight-cell one, because its cell-0 and cell-9 elites would then sit there
    # unable to reproduce with nothing in the output saying so. Recorded as the
    # set, not as the switch, so it stays meaningful if the rule changes again.
    "reproducing_cells",
)

#: What a stored cell SCORE means. The keys still line up, but the numbers
#: defending them came off another budget or another fitness, and comparing
#: those numbers is the archive's whole job.
_SCHEMA_PROTOCOL_FIELDS = (
    "cell_score_formula",
    "cell_measurement",
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip()


def _env_cell_score_formula() -> str:
    """The fitness a cell score is, as this run's environment resolves it.

    Same switches the evaluator reads (lps_graftgate/evaluator_layout_bp_osd),
    so the checkpoint records the formula that actually produced the numbers
    rather than a comment that can drift away from it.
    """
    kdn = _env("LAYOUT_FITNESS_KDN", "0")
    if kdn == "1":
        formula = "k*d^2/n"
    elif kdn in ("d", "D"):
        formula = "d"
    elif kdn in ("", "0"):
        formula = "score_mode=%s" % (_env("LAYOUT_SCORE_MODE", "logeps_kd") or "logeps_kd")
    else:
        formula = "LAYOUT_FITNESS_KDN=%s" % kdn
    return "%s; per-weight best instance (max over cells = combined_score)" % formula


def _env_cell_measurement() -> str:
    """The protocol a cell score was measured under: screen budget -> deep
    budget, and the decode trial count. Two archives whose cells were read at
    8 s and at 16 s hold numbers that are not comparable."""
    return "screen@%ss -> deep@%ss, trials=%s" % (
        _env("LAYOUT_BDIRECT_BUDGET", "1") or "1",
        _env("LAYOUT_BDIRECT_DEEP", "8") or "8",
        _env("LAYOUT_TRIALS", "1") or "1",
    )


def _safe_sum_metrics(metrics: Dict[str, Any]) -> float:
    """Safely sum only numeric metric values, ignoring strings and other types"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) if numeric_values else 0.0


def _safe_avg_metrics(metrics: Dict[str, Any]) -> float:
    """Safely calculate average of only numeric metric values"""
    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) / max(1, len(numeric_values)) if numeric_values else 0.0


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Artifact storage
    artifacts_json: Optional[str] = None  # JSON-serialized small artifacts
    artifact_dir: Optional[str] = None  # Path to large artifact files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


def build_lineage_chain(
    programs: Dict[str, Program],
    start_id: Optional[str],
    max_len: int,
) -> List[Program]:
    """Walk parent_id backwards from start_id and return that program's ANCESTORS.

    Returns at most max_len ancestors in CHRONOLOGICAL order: oldest first, the
    direct parent of start_id last. The program identified by start_id is NOT
    included - it is the "current program" of the prompt and is printed there in
    full.

    This is the real history of one lineage. It is deliberately not the island
    leaderboard: "Previous Attempts" used to be the island's top-N by
    combined_score, so the section showed unrelated programs (often from other
    construction families) and the fitness delta was computed against the
    island's third-best program instead of the parent.

    Returns [] when the chain cannot be reconstructed (no parent_id, parent
    missing from `programs`, e.g. a seed program or a pruned/migrated ancestor).
    Callers must treat [] as "no history", never pad it with other programs.
    """
    chain: List[Program] = []
    if max_len <= 0 or not start_id or start_id not in programs:
        return chain

    seen = {start_id}  # cycle guard: a corrupted parent_id must not hang a worker
    node = programs[start_id]
    while len(chain) < max_len:
        parent_id = node.parent_id
        if not parent_id or parent_id in seen or parent_id not in programs:
            break
        seen.add(parent_id)
        node = programs[parent_id]
        chain.append(node)

    chain.reverse()  # oldest ancestor first, direct parent last
    return chain


class ProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements a combination of MAP-Elites algorithm and
    island-based population model to maintain diversity during evolution.
    It also tracks the absolute best program separately to ensure it's never lost.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Feature grid for MAP-Elites
        self.feature_map: Dict[str, str] = {}

        # Grid width, per dimension (resolve_feature_bins): a configured width
        # for a pre-binned coordinate is exact and is honoured verbatim, only a
        # continuous one may be widened to hold the archive. self.feature_bins
        # is the scalar fallback for a dimension feature_bins_per_dim does not
        # name; consumers that care about the real cell count read
        # _bins_for_dimension / total_cells.
        self.feature_bins, self.feature_bins_per_dim = resolve_feature_bins(config)

        # Bin-islands has a defined answer only for the coordinates listed in
        # BIN_ISLAND_DIMENSIONS. Checked here, at construction, so a run
        # configured with a dimension nobody has written a rule for dies before
        # it spends an LLM call - not silently in whichever direction the last
        # `if` happened to fall.
        if bin_islands_enabled():
            self._bin_islands_dimension()

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Island management attributes
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)  # Default to 10
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)  # Default to 0.1

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track best program per island for proper island-based evolution
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # The MAP-Elites cell this round's parent was drawn from (or the cell
        # requested by the bin-islands path before sampling). None when the
        # parent did not come from a cell.
        self.bin_target: Optional[str] = None

        # Set by _purge_grid when the dropped cells can be recomputed from the
        # kept programs; load() acts on it once the programs are in memory.
        self._grid_rebuild_pending: bool = False

        # Why the last load() dropped the stored grid, None when it kept it.
        # The database DROPS what it cannot honour - it is the lower layer and
        # must never hand a caller a grid in two coordinate systems - but the
        # POLICY (stop the run, or go on with the grid dropped) belongs to the
        # launcher, which cannot see a purge that only wrote a log line. See
        # _purge_grid and lps_graftgate/main.py's _guard_collapsed_grid_resume.
        self.grid_purge_reason: Optional[str] = None

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Prompt log
        self.prompts_by_program: Dict[str, Dict[str, Dict[str, str]]] = None

        # Set random seed for reproducible sampling if specified
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

        # Diversity caching infrastructure
        self.diversity_cache: Dict[int, Dict[str, Union[float, float]]] = (
            {}
        )  # hash -> {"value": float, "timestamp": float}
        self.diversity_cache_size: int = 1000  # LRU cache size
        self.diversity_reference_set: List[str] = (
            []
        )  # Reference program codes for consistent diversity
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)

        # Feature scaling infrastructure
        self.feature_stats: Dict[str, Dict[str, Union[float, float, List[float]]]] = {}
        self.feature_scaling_method: str = "minmax"  # Options: minmax, zscore, percentile

        # feature_bins_per_dim is resolved at the top of __init__, before load()
        # runs: the schema check writes and compares the grid's width, so the
        # width has to exist by then.

        logger.info(f"Initialized program database with {len(self.programs)} programs")

    def add(
        self, program: Program, iteration: int = None, target_island: Optional[int] = None
    ) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add
            iteration: Current iteration (defaults to last_iteration)
            target_island: Specific island to add to (auto-detects parent's island if None)

        Returns:
            Program ID
        """
        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            program.iteration_found = iteration
            # Update last_iteration if needed
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        self._register_on_grid(program)

        # Determine target island
        # Bin-islands: a program belongs to the island of its OWN bin, so a spillover
        # child changes island automatically and island bookkeeping mirrors the frontier.
        # The bin is the ACTIVE feature dimension's metric (weight_bin on this run's grid),
        # not the rate one: reading rate_bin on a weight grid would island programs by a
        # coordinate the archive does not use.
        _bin_metric = self._bin_dimension()
        if (target_island is None and bin_islands_enabled()
                and _bin_metric is not None
                and isinstance(getattr(program, "metrics", None), dict)
                and program.metrics.get(_bin_metric) is not None):
            try:
                target_island = int(float(program.metrics[_bin_metric])) % len(self.islands)
            except (TypeError, ValueError):
                pass
        # If target_island is not specified and program has a parent, inherit parent's island
        if target_island is None and program.parent_id:
            parent = self.programs.get(program.parent_id)
            if parent and "island" in parent.metadata:
                # Child inherits parent's island to maintain island isolation
                island_idx = parent.metadata["island"]
                logger.debug(
                    f"Program {program.id} inheriting island {island_idx} from parent {program.parent_id}"
                )
            else:
                # Parent not found or has no island, use current_island
                island_idx = self.current_island
                if parent:
                    logger.warning(
                        f"Parent {program.parent_id} has no island metadata, using current_island {island_idx}"
                    )
                else:
                    logger.warning(
                        f"Parent {program.parent_id} not found, using current_island {island_idx}"
                    )
        elif target_island is not None:
            # Explicit target island specified (e.g., for migrants)
            island_idx = target_island
        else:
            # No parent and no target specified, use current island
            island_idx = self.current_island

        island_idx = island_idx % len(self.islands)  # Ensure valid island
        self.islands[island_idx].add(program.id)

        # Track which island this program belongs to
        program.metadata["island"] = island_idx

        # Update archive
        self._update_archive(program)

        # Enforce population size limit BEFORE updating best program tracking
        # This ensures newly added programs aren't immediately removed
        self._enforce_population_limit(exclude_program_id=program.id)

        # Update the absolute best program tracking (after population enforcement)
        self._update_best_program(program)

        # Update island-specific best program tracking
        self._update_island_best_program(program, island_idx)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to island {island_idx}")

        return program.id

    def _register_on_grid(self, program: Program) -> None:
        """Put one program in every MAP-Elites cell it is entitled to.

        Everything add() does to the grid and nothing it does to the islands,
        the population cap or the disk, so an archive can also be rebuilt from a
        population that is already in memory without replaying any of that
        bookkeeping (_rebuild_grid_from_programs).
        """
        if not self.config.feature_dimensions:
            return

        feature_coords = self._calculate_feature_coords(program)

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        coords_dict = {
            self.config.feature_dimensions[i]: feature_coords[i]
            for i in range(len(feature_coords))
        }
        self._register_in_cell(program, feature_key, coords_dict)

        # MULTI-CELL REGISTRATION (WEIGHT_ELITES "One policy occupies several
        # cells"). One policy emits 20 instances whose weights differ; keeping
        # only the winner's weight leaves the light cells empty forever, so the
        # archive would preserve nothing the fitness does not already prefer.
        # The program's own coordinate above is untouched - this only ADDS the
        # cells where it holds the best instance seen so far.
        for _w, _wkey in self._weight_cells_of(program):
            if _wkey == feature_key:
                continue
            self._register_in_cell(program, _wkey, {WEIGHT_DIM: _wkey})

    def _rebuild_grid_from_programs(self) -> None:
        """Re-derive feature_map / cell_pools from the loaded population.

        Only ever called after a purge whose reason was the COORDINATES: the
        stored numbers still mean what this run means by them, and each program
        carries the metrics its cells are computed from, so the archive can be
        put back without the 300 re-evaluations a resume never performs. The
        registration is the ordinary one, so the same eligibility rules apply -
        a program that built no instance at a weight does not get that cell here
        either.
        """
        for program in self.programs.values():
            self._register_on_grid(program)
        logger.warning(
            "Rebuilt the MAP-Elites grid from %d loaded program(s): %d cell(s) occupied.",
            len(self.programs),
            len(self.feature_map),
        )

    def get(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID

        Args:
            program_id: Program ID

        Returns:
            Program or None if not found
        """
        return self.programs.get(program_id)

    def sample(self, num_inspirations: Optional[int] = None) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations for the next evolution step

        Args:
            num_inspirations: Number of inspiration programs to sample (defaults to 5 for backward compatibility)

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Select parent program
        parent = self._sample_parent()

        # Select inspirations
        if num_inspirations is None:
            num_inspirations = 5  # Default for backward compatibility
        inspirations = self._sample_inspirations(parent, n=num_inspirations)

        logger.debug(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score or average if None)

        Returns:
            Best program or None if database is empty
        """
        if not self.programs:
            return None

        # If no specific metric and we have a tracked best program, return it
        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:
            # Sort by fitness (excluding feature dimensions)
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by fitness score: {sorted_programs[0].id}")

        # Update the best program tracking if we found a better program
        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")

            # Also log the scores to help understand the update
            if (
                old_id
                and old_id in self.programs
                and "combined_score" in self.programs[old_id].metrics
                and "combined_score" in self.programs[self.best_program_id].metrics
            ):
                old_score = self.programs[old_id].metrics["combined_score"]
                new_score = self.programs[self.best_program_id].metrics["combined_score"]
                logger.info(
                    f"Score change: {old_score:.4f} → {new_score:.4f} ({new_score-old_score:+.4f})"
                )

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses average if None)
            island_idx: If specified, only return programs from this island

        Returns:
            List of top programs
        """
        # Validate island_idx parameter
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

        if not self.programs:
            return []

        # Get candidate programs
        if island_idx is not None:
            # Island-specific query
            island_programs = [
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            ]
            candidates = island_programs
        else:
            # Global query
            candidates = list(self.programs.values())

        if not candidates:
            return []

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            # Sort by combined_score if available, otherwise by average of all numeric metrics
            sorted_programs = sorted(
                candidates,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )

        return sorted_programs[:n]

    def get_lineage(self, program_id: str, n: int = 3) -> List[Program]:
        """Get up to n ancestors of program_id, chronological (direct parent last).

        Thin wrapper over build_lineage_chain so callers holding a live database
        do not have to reach into self.programs. Returns [] if the chain cannot
        be reconstructed.
        """
        return build_lineage_chain(self.programs, program_id, n)

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Perform artifact cleanup before saving
        self._cleanup_old_artifacts(save_path)

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and program.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            # RESUME SAFETY (WEIGHT_ELITES): the keys of feature_map / cell_pools
            # are integers whose MEANING is the grid's dimension, and the values
            # defending those keys are numbers produced by one measurement
            # protocol. The name alone is not enough - two runs can both say
            # weight_bin/9 and mean w=6..14 vs w=5..13, or a cell score read at
            # 8 s vs 16 s. The whole contract is written out so load() can name
            # the field that differs.
            "archive_schema": self._archive_schema(),
            # Kept at the top level as well: written before archive_schema
            # existed, and read by tooling outside this file.
            "feature_dimensions": list(self.config.feature_dimensions or []),
            # The width in force, not the configured one and not a widened one:
            # a reader of this file (lps_graftgate/main.py's stale-cell guard,
            # any offline tool) has to be able to tell which keys the grid can
            # produce at all.
            "feature_bins": self.feature_bins,
            "feature_bins_per_dim": {
                dim: self._bins_for_dimension(dim)
                for dim in list(self.config.feature_dimensions or [])
            },
            "feature_map": self.feature_map,
            # Persisted so a resume does not silently drop back to uniform-over-elites: the
            # sampler requires the pools to cover the grid, and an unsaved dict would fail that
            # test for the rest of the run with no log line saying so.
            "cell_pools": getattr(self, "cell_pools", {}),
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "feature_stats": self._serialize_feature_stats(),
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # One load, one verdict: a database object that is loaded twice must not
        # report the first load's mismatch the second time.
        self.grid_purge_reason = None
        self._grid_rebuild_pending = False

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_map = metadata.get("feature_map", {})
            self.cell_pools = metadata.get("cell_pools", {})
            self._check_grid_dimension(metadata, path)
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self.last_migration_generation = metadata.get("last_migration_generation", 0)

            # Load feature_stats for MAP-Elites grid stability
            self.feature_stats = self._deserialize_feature_stats(metadata.get("feature_stats", {}))

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")
            if self.feature_stats:
                logger.info(f"Loaded feature_stats for {len(self.feature_stats)} dimensions")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        # Reconstruct island assignments from metadata
        self._reconstruct_islands(saved_islands)

        # After the islands, so a rebuilt cell is defended by a program the
        # population actually still holds.
        if self._grid_rebuild_pending:
            self._rebuild_grid_from_programs()

        # Ensure island_generations list has correct length
        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)

        # Ensure island_best_programs list has correct length
        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

        # Log the reconstructed island status
        self.log_island_status()

    def _archive_schema(self) -> Dict[str, Any]:
        """The full contract a stored grid has to satisfy to be resumable.

        Everything a cell KEY or a cell SCORE depends on, resolved from the live
        config and environment - the same resolution the evaluator uses - so a
        resume compares contracts instead of guessing from integers. Recording
        only the dimension name was not enough: two runs can both say
        weight_bin / 9 cells and mean w=6..14 vs w=5..13, or a cell score read
        at 8 s vs one read at 16 s, or k*d^2/n vs k*d^2/n^2.
        """
        dims = list(self.config.feature_dimensions or [])
        schema: Dict[str, Any] = {
            # Recorded for forensics only. The check compares the FIELDS below,
            # never this number, so a version bump on its own never blocks a
            # resume and an unknown (newer) version is accepted as long as the
            # fields it shares with this one agree.
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "feature_dimensions": dims,
            # The width actually IN FORCE, per dimension and as a total. Before
            # resolve_feature_bins these recorded a widened number the grid
            # never used (9 configured, 20 recorded).
            "feature_bins": self.feature_bins,
            "feature_bins_per_dim": {d: self._bins_for_dimension(d) for d in dims},
            "cell_key": "'-'.join(feature coordinates)",
        }
        if dims and all(is_discrete_dimension(d) for d in dims):
            # Every axis is pre-binned, so the cell count is part of what a key
            # MEANS and is compared on resume. A checkpoint written before this
            # field existed simply does not carry it and is not blocked by it.
            schema["cell_count"] = self.total_cells()
        if self._is_rate_grid() and bin_islands_enabled():
            schema["reproducing_cells"] = sorted(
                int(k) for k in self.reproducing_cell_keys(
                    [str(b) for b in range(self._bins_for_dimension("rate_bin"))]))
        if self._is_weight_grid():
            schema.update(
                {
                    "feature_definition": "max row weight over HX and HZ",
                    "feature_range": [WEIGHT_MIN, WEIGHT_MAX],
                    "feature_mapping": "bin = clamp(w, %d, %d) - %d"
                    % (WEIGHT_MIN, WEIGHT_MAX, WEIGHT_MIN),
                    "cell_metric_prefix": WEIGHT_METRIC_PREFIX,
                    "cell_detail_prefixes": list(WEIGHT_DETAIL_PREFIXES),
                    "cell_score_formula": _env_cell_score_formula(),
                    "cell_measurement": _env_cell_measurement(),
                }
            )
        return schema

    def _check_grid_dimension(self, metadata: Dict[str, Any], path: str) -> None:
        """Refuse a checkpoint whose grid means something else than this one's.

        feature_map keys are bare integers. Under [rate_bin] key "9" is a rate
        band; under [weight_bin] it is a row weight. Loading one into the other
        produces an archive in two coordinate systems, where the imported cells
        can never be contested honestly again - and nothing in the numbers says
        so, which is why the whole schema is stored.

        WHERE THE LINE IS DRAWN. The check compares named FIELDS, in two
        classes, and never the schema version:

          COORDINATE fields (feature_dimensions, feature_definition,
            feature_range, feature_mapping, cell_metric_prefix, and cell_count
            on an all-discrete grid) decide what a stored KEY means. A
            difference makes every stored key a different coordinate, so the
            grid cannot be reused.
          PROTOCOL fields (cell_score_formula, cell_measurement) decide what a
            stored SCORE means. The keys still line up, but the numbers
            defending them came off a different budget or a different fitness
            and cannot be compared with the ones this run will produce - and
            comparing them is the archive's whole job.

        Both classes purge by default. Everything else - schema_version,
        feature_bins / feature_bins_per_dim on a grid with a continuous axis,
        cell_detail_prefixes, any field a future version adds and this one does
        not know - is recorded and ignored, so a harmless version bump, a new
        descriptive field, or an archive_size change that widens a continuous
        axis never blocks a resume. A field one side
        has and the other does not is likewise skipped rather than treated as a
        mismatch. LAYOUT_RESUME_LAX=1 downgrades a PROTOCOL mismatch to a
        warning (the coordinates are still honest, you are choosing to compare
        numbers off two budgets); it does not affect COORDINATE mismatches.

        Action on a mismatch is to purge the grid (feature_map / cell_pools) and
        keep the programs: the population is still valid, only its coordinates
        are not. That is what the DATABASE does; whether the RUN continues is
        the launcher's call, and _purge_grid records the reason for it.
        LAYOUT_RESUME_STRICT=1 refuses the resume here instead. A checkpoint
        written before these fields existed is unlabelled; it is treated as
        mismatching whenever the configured grid is the weight grid, because
        every such checkpoint predates the weight grid.
        """
        configured = list(self.config.feature_dimensions or [])
        stored_schema = metadata.get("archive_schema")
        if not isinstance(stored_schema, dict):
            stored_schema = None
        stored_dims = metadata.get("feature_dimensions")
        if stored_schema is not None and "feature_dimensions" in stored_schema:
            stored_dims = stored_schema["feature_dimensions"]

        if stored_schema is None and stored_dims is None:
            if not self._is_weight_grid():
                return  # unlabelled and not a weight run: nothing to compare
            self._purge_grid(
                path,
                "it is unlabelled (written before the archive schema was recorded), "
                "and every unlabelled checkpoint predates the weight grid",
            )
            return

        mine = self._archive_schema()
        if stored_schema is None:
            # Pre-schema checkpoint that still carries the dimension list: the
            # one field it does have is the one that matters most.
            stored_schema = {"feature_dimensions": list(stored_dims or [])}

        # Whether the two sides measured their cell scores the same way, asked
        # independently of which field tripped the check: a coordinate mismatch
        # returns before the protocol fields are reached, and the answer decides
        # whether the purged cells can be rebuilt from the stored metrics.
        # Absent on either side is not agreement - an archive that does not say
        # what budget its numbers came off cannot have them re-used.
        protocol_agrees = bool(_SCHEMA_PROTOCOL_FIELDS) and all(
            field in stored_schema and field in mine and stored_schema[field] == mine[field]
            for field in _SCHEMA_PROTOCOL_FIELDS
        )

        for field_name in _SCHEMA_COORDINATE_FIELDS + _SCHEMA_PROTOCOL_FIELDS:
            if field_name not in stored_schema or field_name not in mine:
                continue
            if stored_schema[field_name] == mine[field_name]:
                continue
            protocol = field_name in _SCHEMA_PROTOCOL_FIELDS
            reason = "%s differs: checkpoint has %r, this run is configured for %r" % (
                field_name,
                stored_schema[field_name],
                mine[field_name],
            )
            if protocol and os.environ.get("LAYOUT_RESUME_LAX", "0") == "1":
                logger.warning(
                    "checkpoint %s: %s. LAYOUT_RESUME_LAX=1, keeping the grid - "
                    "its cell scores were measured under the other protocol.",
                    path,
                    reason,
                )
                continue
            self._purge_grid(path, reason, scores_comparable=not protocol and protocol_agrees)
            return

        if list(stored_dims or []) != configured:
            self._purge_grid(
                path,
                "feature_dimensions differs: checkpoint has %r, this run is "
                "configured for %r" % (list(stored_dims or []), configured),
                scores_comparable=protocol_agrees,
            )

    def _purge_grid(self, path: str, reason: str, scores_comparable: bool = False) -> None:
        """Drop the stored grid, name the field that did it, tell the caller.

        The keys go and the programs stay, but what the RUN does about it is not
        decided here: `grid_purge_reason` carries the mismatch up to whoever
        loaded the checkpoint (lps_graftgate/main.py stops the resume unless the
        operator asked for the purge). A purge that only wrote a log line was
        invisible to the launcher's own guard, which then found an empty grid
        and passed. LAYOUT_RESUME_STRICT=1 refuses here instead, for callers
        that have no launcher above them.

        The cells come BACK, rebuilt from the kept programs' own metrics, when
        the stored numbers are still this run's - it was the coordinates that
        were wrong, not the measurements, and load() does not re-evaluate
        anything, so nothing else would ever restore them. When the measurement
        protocol itself differs the rebuild is the one thing that must not
        happen: those numbers came off another budget and cannot defend a cell
        against the ones this run will produce.
        """
        cells = len(self.feature_map or {}) + len(getattr(self, "cell_pools", None) or {})
        message = (
            "checkpoint %s does not match this run's archive schema - %s. "
            "Its %d stored cell key(s)/pool(s) cannot be contested honestly here."
            % (path, reason, cells)
        )
        if os.environ.get("LAYOUT_RESUME_STRICT", "0") == "1":
            raise ValueError(
                "REFUSING TO RESUME: "
                + message
                + " Start clean with --force-clear, or unset LAYOUT_RESUME_STRICT to "
                "resume with the stored grid purged (programs are kept)."
            )
        self.feature_map = {}
        self.cell_pools = {}
        self.grid_purge_reason = reason
        # A coordinate is recomputable from a program's own metrics only while
        # every axis is a pre-binned metric. A continuous axis is scaled against
        # feature_stats that move as programs are processed, so a rebuild there
        # would invent coordinates that depend on the order the checkpoint
        # happened to be read in.
        dims = list(self.config.feature_dimensions or [])
        self._grid_rebuild_pending = bool(
            scores_comparable and dims and all(is_discrete_dimension(d) for d in dims)
        )
        logger.warning(
            "%s Purging feature_map and cell_pools; %s",
            message,
            "the programs are kept and their cells are rebuilt from the metrics "
            "those programs already carry."
            if self._grid_rebuild_pending
            else "the programs are kept but their cells are not - load() does not "
            "re-evaluate, so the grid refills only as new children arrive.",
        )

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island program ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_programs = []
        restored_programs = 0

        # Restore island assignments
        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for program_id in program_ids:
                if program_id in self.programs:
                    # Program exists, add to island
                    self.islands[island_idx].add(program_id)
                    # Set island metadata on the program
                    self.programs[program_id].metadata["island"] = island_idx
                    restored_programs += 1
                else:
                    # Program missing, track it
                    missing_programs.append((island_idx, program_id))

        # Clean up archive - remove missing programs
        original_archive_size = len(self.archive)
        self.archive = {pid for pid in self.archive if pid in self.programs}

        # Clean up feature_map - remove missing programs
        feature_keys_to_remove = []
        for key, program_id in self.feature_map.items():
            if program_id not in self.programs:
                feature_keys_to_remove.append(key)
        for key in feature_keys_to_remove:
            del self.feature_map[key]

        # Clean up island best programs - remove stale references
        self._cleanup_stale_island_bests()

        # Check best program
        if self.best_program_id and self.best_program_id not in self.programs:
            logger.warning(f"Best program {self.best_program_id} not found, will recalculate")
            self.best_program_id = None

        # Log reconstruction results
        if missing_programs:
            logger.warning(
                f"Found {len(missing_programs)} missing programs during island reconstruction:"
            )
            for island_idx, program_id in missing_programs[:5]:  # Show first 5
                logger.warning(f"  Island {island_idx}: {program_id}")
            if len(missing_programs) > 5:
                logger.warning(f"  ... and {len(missing_programs) - 5} more")

        if original_archive_size > len(self.archive):
            logger.info(
                f"Removed {original_archive_size - len(self.archive)} missing programs from archive"
            )

        if feature_keys_to_remove:
            logger.info(f"Removed {len(feature_keys_to_remove)} missing programs from feature map")

        logger.info(f"Reconstructed islands: restored {restored_programs} programs to islands")

        # If we have programs but no island assignments, distribute them
        if self.programs and sum(len(island) for island in self.islands) == 0:
            logger.info("No island assignments found, distributing programs across islands")
            self._distribute_programs_to_islands()

    def _distribute_programs_to_islands(self) -> None:
        """
        Distribute loaded programs across islands when no island metadata exists
        """
        program_ids = list(self.programs.keys())

        # Distribute programs round-robin across islands
        for i, program_id in enumerate(program_ids):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(program_id)
            self.programs[program_id].metadata["island"] = island_idx

        logger.info(f"Distributed {len(program_ids)} programs across {len(self.islands)} islands")

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _calculate_feature_coords(self, program: Program) -> List[int]:
        """
        Calculate feature coordinates for the MAP-Elites grid

        Args:
            program: Program to calculate features for

        Returns:
            List of feature coordinates
        """
        coords = []
        if not self.config.feature_dimensions:
            return coords

        for dim in self.config.feature_dimensions:
            if dim == "complexity":
                # Use code length as complexity measure
                complexity = len(program.code)
                bin_idx = self._calculate_complexity_bin(complexity)
                coords.append(bin_idx)
            elif dim == "diversity":
                # Use cached diversity calculation with reference set
                if len(self.programs) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(program)
                    bin_idx = self._calculate_diversity_bin(diversity)
                coords.append(bin_idx)
            elif dim == "score":
                # Use average of numeric metrics
                if not program.metrics:
                    bin_idx = 0
                else:
                    # Use fitness score for "score" dimension (consistent with rest of system)
                    avg_score = get_fitness_score(program.metrics, self.config.feature_dimensions)
                    # Update stats and scale
                    self._update_feature_stats("score", avg_score)
                    scaled_value = self._scale_feature_value("score", avg_score)
                    num_bins = self._bins_for_dimension("score")
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim in program.metrics:
                # Use specific metric
                score = program.metrics[dim]
                num_bins = self._bins_for_dimension(dim)
                if dim.endswith("_bin"):
                    # Convention: a *_bin metric IS the bin index (fixed edges
                    # computed by the evaluator) - no adaptive scaling.
                    bin_idx = max(0, min(num_bins - 1, int(score)))
                    coords.append(bin_idx)
                    continue
                # Update stats and scale
                self._update_feature_stats(dim, score)
                scaled_value = self._scale_feature_value(dim, score)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim.endswith("_bin"):
                # *_bin dim missing => failed evaluation (reduced metrics dict);
                # park it in bin 0 so the program still enters the population.
                coords.append(0)
            else:
                # Feature not found - this is an error
                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )
        # Only log coordinates at debug level for troubleshooting
        logger.debug(
            "MAP-Elites coords: %s",
            str({self.config.feature_dimensions[i]: coords[i] for i in range(len(coords))}),
        )
        return coords

    def _calculate_complexity_bin(self, complexity: int) -> int:
        """
        Calculate the bin index for a given complexity value using feature scaling.

        Args:
            complexity: The complexity value (code length)

        Returns:
            Bin index in range [0, this dimension's own bin count - 1]
        """
        # Update feature statistics
        self._update_feature_stats("complexity", float(complexity))

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("complexity", float(complexity))

        # Get number of bins for this dimension
        num_bins = self._bins_for_dimension("complexity")

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _calculate_diversity_bin(self, diversity: float) -> int:
        """
        Calculate the bin index for a given diversity value using feature scaling.

        Args:
            diversity: The average fast code diversity to other programs

        Returns:
            Bin index in range [0, this dimension's own bin count - 1]
        """
        # Update feature statistics
        self._update_feature_stats("diversity", diversity)

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("diversity", diversity)

        # Get number of bins for this dimension
        num_bins = self._bins_for_dimension("diversity")

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates

        Returns:
            String key
        """
        return "-".join(str(c) for c in coords)

    # ------------------------------------------------------------------
    # Grid width (resolve_feature_bins)
    # ------------------------------------------------------------------

    def _bins_for_dimension(self, dim: str) -> int:
        """How many cells the grid has along `dim`.

        The per-dimension width is authoritative; self.feature_bins is only the
        fallback for a dimension no configuration named. Every binning and
        clamping decision reads this, so no caller can disagree with the grid
        about how wide one axis is.
        """
        try:
            return max(1, int(self.feature_bins_per_dim.get(dim, self.feature_bins)))
        except (TypeError, ValueError):
            return max(1, int(self.feature_bins or 1))

    def total_cells(self) -> int:
        """How many cells the CONFIGURED grid has: the coverage denominator.

        The product over dimensions of each dimension's own width, never
        feature_bins ** ndim - the two differ as soon as one axis is discrete
        and another is not, and the coverage percentage is a number the run is
        judged by.
        """
        total = 1
        for dim in list(self.config.feature_dimensions or []):
            total *= self._bins_for_dimension(dim)
        return max(1, total)

    # ------------------------------------------------------------------
    # Weight cells (WEIGHT_ELITES.md)
    # ------------------------------------------------------------------

    def _is_weight_grid(self) -> bool:
        """True when the configured grid is the one-dimensional weight grid.

        Everything weight-specific below is inert on any other grid, so a rate
        grid, a multi-dimensional grid and a run with no grid at all keep the
        behaviour they had before this change.
        """
        return list(self.config.feature_dimensions or []) == [WEIGHT_DIM]

    def _bin_dimension(self) -> Optional[str]:
        """The metric name of the single active bin dimension, or None.

        Bin-islands is a one-dimensional idea (one island per cell of one axis),
        so it engages only on a one-dimensional grid. Returning the CONFIGURED
        name is what keeps the rate special cases from leaking onto the weight
        grid.
        """
        dims = list(self.config.feature_dimensions or [])
        return dims[0] if len(dims) == 1 else None

    def _bin_islands_dimension(self) -> str:
        """The coordinate bin-islands is running on, or raise.

        Asking this at all means LAYOUT_BIN_ISLANDS=1, and bin-islands only has
        a defined meaning for the coordinates in BIN_ISLAND_DIMENSIONS: it has
        to decide which cells may parent a child, and that answer comes from
        what the coordinate's edges ARE. On an unknown dimension both answers
        are wrong in different ways - applying the rate rule sterilises two
        legitimate corners, ignoring it lets out-of-window bins reproduce - and
        either way nothing in the run's output would say which one it picked.
        """
        dims = list(self.config.feature_dimensions or [])
        dim = dims[0] if len(dims) == 1 else None
        if dim not in BIN_ISLAND_DIMENSIONS:
            raise ValueError(
                "LAYOUT_BIN_ISLANDS=1 has no defined behaviour for "
                "feature_dimensions=%r. Which cells may parent a child is a "
                "property of the coordinate (%s), and bin-islands is a "
                "one-dimensional mechanism. Either configure one of those "
                "dimensions, add an explicit rule for this one in "
                "reproducing_cell_keys, or set LAYOUT_BIN_ISLANDS=0."
                % (dims, ", ".join(BIN_ISLAND_DIMENSIONS))
            )
        return dim

    def _weight_bin_offset(self) -> int:
        """Weight of cell 0: WEIGHT_MIN, the coordinate the evaluator publishes
        (weight_bin = clamp(max row weight, 6, 14) - 6, nine cells 0..8).

        Deliberately NOT derived from the grid's width. The width is a
        configuration number and the offset is part of what a KEY MEANS; an
        offset read off the width would alias cell 6 (weight 12) with weight 6
        the moment the two disagreed - which they did, before
        resolve_feature_bins stopped widening a discrete dimension.
        """
        return WEIGHT_MIN

    def _cell_weight(self, feature_key: Optional[str]) -> Optional[int]:
        """The max row weight the cell `feature_key` stands for, or None."""
        if feature_key is None or not self._is_weight_grid():
            return None
        try:
            bin_idx = int(str(feature_key))
        except (TypeError, ValueError):
            return None
        weight = bin_idx + self._weight_bin_offset()
        return weight if WEIGHT_MIN <= weight <= WEIGHT_MAX else None

    def _weight_cell_key(self, weight: int) -> str:
        """The feature-map key of the cell that holds max row weight `weight`."""
        return str(int(weight) - self._weight_bin_offset())

    def _weight_cells_of(self, program: Program) -> List[Tuple[int, str]]:
        """[(weight, cell key)] for every w_best_<w> this program carries."""
        metrics = getattr(program, "metrics", None) or {}
        if not self._is_weight_grid():
            # Evidence, not configuration: if the evaluator is emitting weight
            # elites and the grid is not the weight grid, the whole mechanism is
            # silently inert, which is exactly the failure mode that has bitten
            # this project before. Say so once.
            if not getattr(self, "_warned_weight_grid_off", False) and any(
                str(k).startswith(WEIGHT_METRIC_PREFIX) for k in metrics
            ):
                self._warned_weight_grid_off = True
                logger.warning(
                    "Programs carry %s* metrics but feature_dimensions is %s, not ['%s']: "
                    "multi-cell weight registration is OFF for this run.",
                    WEIGHT_METRIC_PREFIX,
                    list(self.config.feature_dimensions or []),
                    WEIGHT_DIM,
                )
            return []
        cells = []
        for weight in range(WEIGHT_MIN, WEIGHT_MAX + 1):
            if weight_score(metrics, weight) is not None:
                cells.append((weight, self._weight_cell_key(weight)))
        return cells

    def _cell_score(self, program: Program, feature_key: Optional[str]) -> Optional[float]:
        """The score that decides THIS cell, or None when the program is NOT
        ELIGIBLE for it.

        Four defined cases, in this order:

          1. weight grid, program carries w_best_<w> for this cell's weight
             -> that deep-verified number. This is the only number that may
             compete for the cell.
          2. weight grid, program carries w_best_* for OTHER weights but not
             this one -> None. It is under the weight schema and it built no
             instance here, so it is not a candidate for this cell at all.
             Falling back to the global fitness here is what let a heavy global
             winner take (and pollute) a light cell it never built in:
             combined_score is the MAX over the cells it DID occupy.
          3. no w_best_* anywhere but the program still carries a weight_bin
             coordinate (a program from before the schema) -> the global
             fitness, exactly the comparison every cell used before this
             change. Such a program has claimed no cell, so its fitness is not
             a per-cell maximum and cannot smuggle another cell's number in
             here. _cell_rank still ranks it BELOW every schema-scored program,
             so the first properly measured challenger takes the cell off it.
          4. no w_best_* AND no weight_bin (a failed evaluation with a reduced
             metrics dict) -> None. The evaluator placed it nowhere on the
             weight axis, so it is an elite of no weight. Without this it would
             be parked in bin 0 by _calculate_feature_coords' missing-*_bin
             fallback and would squat on the w=6 cell, which is the lightest
             corner and the one hardest to fill honestly.

        Off the weight grid, and for a key outside the nine cells, the answer is
        the global fitness in every case: nothing here changes a rate grid, a
        multi-dimensional grid or a run with no grid at all.
        """
        weight = self._cell_weight(feature_key)
        metrics = getattr(program, "metrics", None) or {}
        if weight is not None:
            value = weight_score(metrics, weight)
            if value is not None:
                return value
            if any(str(k).startswith(WEIGHT_METRIC_PREFIX) for k in metrics):
                return None  # case 2: under the schema, built nothing here
            if WEIGHT_DIM not in metrics:
                return None  # case 4: no weight information at all
        return get_fitness_score(program.metrics, self.config.feature_dimensions)

    def _program_shape(self, program: Program):
        """(A_SHAPE, B_SHAPE) of a program, or None when it cannot be read.

        The evaluator writes a_shape / b_shape into the metrics; older records
        and programs that failed before evaluation do not have them, so the
        source is parsed as a fallback. Returning None makes the caller keep
        the program in a group of its own rather than merge it with a shape it
        may not have.
        """
        m = getattr(program, "metrics", None) or {}
        a, b = m.get("a_shape"), m.get("b_shape")
        if a is not None and b is not None:
            try:
                return (tuple(int(x) for x in a), tuple(int(x) for x in b))
            except (TypeError, ValueError):
                pass
        import re as _re
        code = getattr(program, "code", "") or ""
        ma = _re.search(r"^A_SHAPE\s*=\s*\((\d+),\s*(\d+)\)", code, _re.M)
        mb = _re.search(r"^B_SHAPE\s*=\s*\((\d+),\s*(\d+)\)", code, _re.M)
        if ma and mb:
            return ((int(ma.group(1)), int(ma.group(2))),
                    (int(mb.group(1)), int(mb.group(2))))
        return None

    def _cell_rank(self, program: Program, feature_key: Optional[str]) -> Tuple[int, float]:
        """Ordering key for one cell: (tier, score), higher is better.

        tier 1  scored by this cell's own measurement (_cell_score case 1)
        tier 0  legacy, ranked by global fitness (case 3), and every cell of a
                non-weight grid
        tier -1 ineligible (cases 2 and 4) - sorts below everything, never wins

        Tuple comparison does the whole job: an ineligible program is never
        better than anything, and a legacy holder yields to the first
        schema-scored challenger regardless of how large its fitness is.
        """
        weight = self._cell_weight(feature_key)
        if weight is not None:
            measured = weight_score(getattr(program, "metrics", None), weight)
            if measured is not None:
                return (1, measured)
        score = self._cell_score(program, feature_key)
        if score is None:
            return (-1, float("-inf"))
        return (0, score)

    def _cell_eligible(self, program: Program, feature_key: Optional[str]) -> bool:
        """False when this program may not hold or contest `feature_key` at all."""
        return self._cell_score(program, feature_key) is not None

    def reproducing_cell_keys(self, keys: List[str]) -> List[str]:
        """Which of `keys` may parent a child under bin-islands.

        This is a property of the ACTIVE feature dimension, not a constant
        (BIN_ISLAND_DIMENSIONS):

        rate_bin: cell 0 and cell LAYOUT_RATE_BIN_N-1 are the out-of-window
            boundary bins the rate scheme deliberately builds - everything below
            the floor lands in one and everything above the ceiling in the
            other - so they hold programs that must not reproduce.

        weight_bin: no such boundary exists. The nine cells are clamp(w,6,14)-6
            and every one of them is a legitimate corner; w=6 and w=14 are the
            EXTREMES this dimension was added to develop (WEIGHT_ELITES: the
            light region was not losing per sample, nobody was sampling it).
            Applying the rate rule here would sterilise them.

        Any other dimension raises: there is no rule to apply and no safe
        default (see _bin_islands_dimension).

        The ONE place that answers this question. The controller has to pick a
        target cell before sampling (process_parallel._submit_iteration) and
        _sample_exploitation_parent has to restrict the draw to the same set;
        the two answering it separately is exactly how the rate rule survived
        on the weight grid inside the controller after the database stopped
        applying it.

        When the rate rule would leave nothing at all (only boundary bins are
        occupied), the full list is returned rather than an empty one: a run
        with no reproducing cell would stop reproducing entirely.
        """
        dim = self._bin_islands_dimension()
        keys = list(keys)
        if dim == WEIGHT_DIM:
            return keys
        # ONE authority for how wide this axis is: the configured per-dimension
        # bin count, which every binning and clamping decision already reads
        # (_bins_for_dimension). This used to be LAYOUT_RATE_BIN_N, default 22,
        # which nothing ever set - so on the 15-cell grid the run actually used,
        # cell 14 (the TOP out-of-window clip bin, where rate_bin_of dumps
        # everything above the ceiling) was treated as interior and allowed to
        # reproduce, which is the one thing this method exists to prevent.
        # LAYOUT_RATE_EDGE_CELLS=1: develop the two boundary
        # cells as well. They remain open-ended - rate_bin_of clamps everything
        # below the floor into cell 0 and everything above the ceiling into the
        # last one - but they are no longer barred from parenting, and what the
        # round is TOLD about them changes with them (cell_target_rate). Unset
        # keeps the original rule, under which those two are buckets only.
        if os.environ.get("LAYOUT_RATE_EDGE_CELLS", "0") == "1":
            return [k for k in keys if str(k).isdigit()] or keys
        n_bins = self._bins_for_dimension(dim)
        interior = [k for k in keys if str(k).isdigit() and 0 < int(k) < n_bins - 1]
        return interior or keys

    def occupied_reproducing_cells(self) -> List[str]:
        """Occupied cells that may parent a child, in coordinate order.

        What the controller round-robins its islands over. Empty cells are not
        in it - they fill by spillover - and with fewer cells than slots the
        extra slots deepen the occupied ones instead of idling.
        """
        allowed = self.reproducing_cell_keys([str(k) for k in self.feature_map.keys()])
        # Numeric keys sort numerically ("10" after "9"), anything else after
        # them by string, so a given island always gets the same cell.
        return sorted(allowed, key=lambda k: (0, int(k), "") if k.isdigit() else (1, 0, k))

    def _update_cell_pool(self, program: Program, feature_key: str) -> None:
        """Per-cell pool: keep the best K programs of this cell, not just its elite, so a
        cell's lineage stays reachable as a parent instead of being discarded the moment
        it is beaten. Off unless LAYOUT_CELL_POOL is set; feature_map is untouched either
        way, so every existing consumer sees the old behaviour.

        Ranked by _cell_rank: a policy whose winner is heavy would otherwise
        crowd the light cells' pools with a number it did not earn there, and a
        policy that built no instance at this weight is not admitted at all.
        """
        try:
            pool_k = int(os.environ.get("LAYOUT_CELL_POOL", "0") or 0)
        except ValueError:
            pool_k = 0
        if pool_k <= 0:
            return
        if not hasattr(self, "cell_pools"):
            self.cell_pools = {}
        pool = [pid for pid in self.cell_pools.get(feature_key, []) if pid in self.programs]
        if program.id not in pool and self._cell_eligible(program, feature_key):
            pool.append(program.id)
        pool.sort(key=lambda pid: self._cell_rank(self.programs[pid], feature_key), reverse=True)
        # SHAPE-STRATIFIED POOL (2026-09-19). Ranking by score alone fills the
        # pool with one shape's consecutive variants (measured: 12 programs,
        # 1 distinct shape in 5 of 8 cells), which leaves the shape-uniform
        # parent draw with a single group. Under LAYOUT_SHAPE_PARENT_K the pool
        # instead holds each shape's own best in this cell, for up to pool_k
        # shapes. Programs whose shape cannot be read keep their own slot.
        try:
            _shape_k = int(os.environ.get("LAYOUT_SHAPE_PARENT_K", "0") or 0)
        except ValueError:
            _shape_k = 0
        if _shape_k > 0:
            _seen = {}
            for _pid in pool:                      # already best-first
                _sh = self._program_shape(self.programs[_pid]) or ("id", _pid)
                if _sh not in _seen:
                    _seen[_sh] = _pid
            pool = list(_seen.values())
        self.cell_pools[feature_key] = pool[:pool_k]

    def _register_in_cell(
        self, program: Program, feature_key: str, coords_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        """MAP-Elites registration of one program in ONE cell.

        Called once for the program's own coordinate and once for every weight
        cell it holds the best instance in. The replacement test is _cell_rank,
        not the global fitness.
        """
        # A program that is not eligible for this cell may not take it, not even
        # when it is empty: on the weight grid an ineligible program is one that
        # is under the weight schema and built no instance at this weight, so
        # letting it occupy the cell would fill the light corner with a number
        # measured somewhere else and stop any honest holder from ever arriving.
        if not self._cell_eligible(program, feature_key):
            return False

        existing_program_id = self.feature_map.get(feature_key)
        should_replace = existing_program_id is None

        if not should_replace:
            # Check if the existing program still exists before comparing
            if existing_program_id not in self.programs:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale program reference {existing_program_id} in feature map"
                )
            else:
                # Program exists, compare on this cell's own score
                should_replace = self._is_better(
                    program, self.programs[existing_program_id], feature_key=feature_key
                )

        if should_replace:
            # Log significant MAP-Elites events
            label = coords_dict if coords_dict is not None else {"cell": feature_key}

            if existing_program_id is None:
                # New cell occupation
                logger.info("New MAP-Elites cell occupied: %s", label)
                # Coverage milestone. The denominator is the real cell count
                # (total_cells), not feature_bins ** ndim: on the nine-cell
                # weight grid the latter reported 20 cells and every coverage
                # figure was 45% of the truth.
                total_possible_cells = self.total_cells()
                occupied = len(self.feature_map) + 1
                coverage = occupied / total_possible_cells
                previous = (occupied - 1) / total_possible_cells
                # Milestones are CROSSED, not hit exactly: with nine cells no
                # occupancy ever equals 0.25 or 0.5, so an equality test logged
                # nothing at all on this grid.
                for milestone in (0.1, 0.25, 0.5, 0.75, 0.9):
                    if previous < milestone <= coverage:
                        logger.info(
                            "MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                            coverage * 100,
                            occupied,
                            total_possible_cells,
                        )
                        break
            elif existing_program_id in self.programs:
                # Cell replacement - existing program being replaced
                logger.info(
                    "MAP-Elites cell improved: %s (cell score: %s -> %s)",
                    label,
                    _fmt_cell_score(
                        self._cell_score(self.programs[existing_program_id], feature_key)
                    ),
                    _fmt_cell_score(self._cell_score(program, feature_key)),
                )

            self.feature_map[feature_key] = program.id

            # ARCHIVE SYNC, REFERENCE COUNTED. One policy holds SEVERAL weight
            # cells, so losing one cell says nothing about whether it is still an
            # elite: the incumbent gives up its archive slot only when the
            # reassignment above left it holding NO cell at all. Evicting on the
            # single-cell event deleted multi-cell programs from the archive
            # while they were still the best thing anywhere else on the grid.
            # Checked AFTER the reassignment so this cell no longer counts, and
            # the slot is handed over rather than added, so archive_size holds:
            # a challenger whose incumbent stays goes through _update_archive on
            # its own merit like any other program.
            if existing_program_id is not None and existing_program_id in self.archive:
                if not self._holds_any_cell(existing_program_id):
                    self.archive.discard(existing_program_id)
                    self.archive.add(program.id)

        self._update_cell_pool(program, feature_key)
        return should_replace

    def _holds_any_cell(self, program_id: str) -> bool:
        """True while `program_id` is the elite of at least one cell.

        The reference count behind archive eviction. "Elite" is elite_ids() and
        nothing else: three consumers that each decided for themselves which
        maps counted is how a program could be protected from the pruner as an
        island elite while the prompt's frontier block could not see it.
        """
        return program_id in self.elite_ids()

    def weight_frontier(self) -> List[Dict[str, Any]]:
        """Best result at each max row weight, run-wide, one record per weight.

        Built from the elites (elite_ids() and the archive), never from one
        policy's metrics: the point of the block is to show the model which
        corners of the weight axis are weak or untouched. Records for weights
        nothing has reached carry score None.

        The archive alone is not enough. _update_archive keeps the archive's
        best by GLOBAL fitness, which correlates with heavy weights, so a light
        cell's elite can be evicted from the archive while still holding its
        cell; the union is what "the elites" actually means here. elite_ids()
        rather than feature_map, so the block shows the same programs the pruner
        protects - under V2_POLICY_BINS a program can hold an island cell only,
        and the model was being shown an empty corner that was in fact taken.
        """
        if not self._is_weight_grid():
            return []

        candidates = self.elite_ids() | set(self.archive)
        frontier: Dict[int, Dict[str, Any]] = {}
        for pid in candidates:
            program = self.programs.get(pid)
            if program is None:
                continue
            metrics = program.metrics or {}
            for weight, _key in self._weight_cells_of(program):
                score = weight_score(metrics, weight)
                if score is None:
                    continue
                best = frontier.get(weight)
                if best is not None and best["score"] >= score:
                    continue
                record = {"weight": weight, "score": score, "program_id": pid}
                record.update(weight_detail(metrics, weight))
                frontier[weight] = record

        return [
            frontier.get(w, {"weight": w, "score": None})
            for w in range(WEIGHT_MIN, WEIGHT_MAX + 1)
        ]

    def cell_target_weight(self) -> Optional[int]:
        """The max row weight of the cell THIS round's parent was drawn from.

        None when the parent did not come from a cell (exploration / random
        parent) or the grid is not the weight grid. bin_target is the drawn cell
        - the attribute the bin-islands path already used to REQUEST one - and
        _sample_exploitation_parent overwrites it with the cell it actually drew.
        """
        return self._cell_weight(getattr(self, "bin_target", None))

    def _is_rate_grid(self) -> bool:
        """True when the configured grid is the one-dimensional rate grid."""
        return list(self.config.feature_dimensions or []) == ["rate_bin"]

    def rate_grid(self) -> Tuple[float, float, int]:
        """(lo, step, n_bins) of the rate grid.

        lo and step come from the evaluator's own CM_RATE_LO / CM_RATE_STEP:
        the band named in a round's prompt has to be the band the evaluator
        bins the child into, and the evaluator computes that from these two.

        n_bins comes from the GRID, not from CM_RATE_BINS, because the grid is
        what decides which cells can exist at all. CM_RATE_BINS is checked
        against it and a disagreement is logged loudly rather than silently
        producing rounds that ask for cells the grid cannot hold.
        """
        lo = float(os.environ.get("CM_RATE_LO", "0.15"))
        step = float(os.environ.get("CM_RATE_STEP", "0.025"))
        n = self._bins_for_dimension("rate_bin")
        env_n = os.environ.get("CM_RATE_BINS")
        if env_n is not None and not getattr(self, "_rate_grid_warned", False):
            try:
                if int(env_n) != int(n):
                    logger.error(
                        "RATE GRID MISMATCH: the evaluator bins with "
                        "CM_RATE_BINS=%s but the MAP-Elites grid has %d cells "
                        "(feature_bins.rate_bin). Rounds will ask for cells the "
                        "grid cannot hold, or cells the evaluator never emits.",
                        env_n, n)
            except (TypeError, ValueError):
                logger.error("RATE GRID: CM_RATE_BINS=%r is not an integer", env_n)
            self._rate_grid_warned = True
        return lo, step, n

    def cell_target_rate(self) -> Optional[Dict[str, Any]]:
        """The rate cell THIS round is asked to fill, or None.

        None unless the controller set cell_target_key for this round, which it
        does only under bin-islands on a rate grid. There is deliberately NO
        fallback to bin_target: bin_target is written by the exploitation
        sampler whether or not bin-islands is on, so a fallback made every
        exploitation round claim to be filling the cell its parent already
        occupies.
        """
        key = getattr(self, "cell_target_key", None)
        if key is None or not self._is_rate_grid():
            return None
        try:
            b = int(str(key))
        except (TypeError, ValueError):
            return None
        lo, step, n = self.rate_grid()
        if not (0 <= b < n):
            return None
        centre = lo + step * b
        # Where the parent REALLY came from, recorded by the controller after
        # every path that can change the parent (sample(), then the v6c stay
        # queue). Not derived from bin_target: under LAYOUT_CELL_POOL that is
        # the drawn cell key, which need not be the parent's own cell.
        try:
            parent_cell = int(str(getattr(self, "parent_cell_key", None)))
        except (TypeError, ValueError):
            parent_cell = None
        return {
            "cell": b,
            "centre": centre,
            "lo": centre - step / 2.0,
            "hi": centre + step / 2.0,
            # The two boundary cells are open-ended by construction: the
            # evaluator clamps every rate below the floor into cell 0 and every
            # rate above the ceiling into the last one. While they were barred
            # from reproducing no round ever named them, so stating a closed
            # band cost nothing; under LAYOUT_RATE_EDGE_CELLS they are named on
            # a fifth of the rounds, and "k/n in [0.5250, 0.5750)" would make
            # the model reject the whole high-rate family the switch exists to
            # develop. Read by prompt.sampler.render_rate_cell_focus.
            "open_lo": b == 0 and os.environ.get("LAYOUT_RATE_EDGE_CELLS", "0") == "1",
            "open_hi": b == n - 1 and os.environ.get("LAYOUT_RATE_EDGE_CELLS", "0") == "1",
            "occupied": str(b) in self.feature_map,
            "parent_cell": parent_cell,
            "parent_centre": (lo + step * parent_cell) if parent_cell is not None else None,
            "n_bins": n,
        }

    def pick_target_cell(self, iteration: int) -> Optional[str]:
        """The cell this round is asked to fill, on a RATE grid.

        Every cell allowed to reproduce, occupied or not - deliberately not
        occupied_reproducing_cells(), which can only deepen cells that already
        hold something, so an empty band is never targeted and fills only if a
        child happens to drift into it. The parent for an empty target comes
        from the nearest occupied cell (_nearest_cell_key).

        Drawn from a SHUFFLED PERMUTATION refreshed every len(allowed) rounds,
        not i.i.d. uniform: "every cell gets the same number of rounds" is a
        claim about counts, and independent draws give a multinomial spread
        (2 to 7 over 60 rounds with 13 cells) while a permutation gives equal
        counts up to the last partial block. Seeded by the block index so a
        resume replays the same sequence.

        Note this is the target, not the round: ~30% of rounds take the
        exploration or random path in _sample_parent, which serve no cell.
        """
        _, _, n = self.rate_grid()
        allowed = self.reproducing_cell_keys([str(b) for b in range(n)])
        if not allowed:
            return None
        # LAYOUT_LOCK_CELL=<b>: every round targets cell b (single-seed lines,
        # 2026-09-14). The cell must be one the grid allows to reproduce;
        # anything else is logged once and ignored, so a typo cannot silently
        # turn the lock into the rotating default.
        _lock = os.environ.get("LAYOUT_LOCK_CELL", "").strip()
        if _lock:
            if _lock in [str(k) for k in allowed]:
                return _lock
            if not getattr(self, "_lock_cell_warned", False):
                self._lock_cell_warned = True
                logger.error("LAYOUT_LOCK_CELL=%r is not a reproducing cell of this grid "
                             "(%s); falling back to the rotating target", _lock, sorted(allowed))
        allowed = sorted(allowed, key=lambda k: (0, int(k), "") if str(k).isdigit()
                         else (1, 0, str(k)))
        m = len(allowed)
        it = int(iteration)
        return random.Random(it // m).sample(allowed, m)[it % m]

    def _nearest_cell_key(self, target, keys) -> Optional[str]:
        """The occupied cell closest to `target`, ties by that cell's score.

        Used when the round's target cell is empty. Nearest rather than best:
        rate is a lattice in shape space, so the neighbouring band is a smaller
        rewrite than the run's strongest cell would be.
        """
        try:
            t = int(str(target))
        except (TypeError, ValueError):
            return None
        cand = [str(k) for k in keys if str(k).isdigit()]
        if not cand:
            return None

        def _score(k):
            pid = self.feature_map.get(k)
            prog = self.programs.get(pid) if pid else None
            metrics = getattr(prog, "metrics", None) or {}
            try:
                return float(metrics.get("combined_score", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        return min(cand, key=lambda k: (abs(int(k) - t), -_score(k), int(k)))

    def _is_better(
        self, program1: Program, program2: Program, feature_key: Optional[str] = None
    ) -> bool:
        """
        Determine if program1 has better FITNESS than program2

        Uses fitness calculation that excludes MAP-Elites feature dimensions
        to prevent pollution of fitness comparisons.

        Args:
            program1: First program
            program2: Second program
            feature_key: when given, the comparison is about THAT cell and uses
                the cell's own score (WEIGHT_ELITES): combined_score is the max
                over a policy's occupied cells, so a heavy winner would take
                every light cell it touches.

        Returns:
            True if program1 is better than program2
        """
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        if feature_key is not None:
            # _cell_rank, not _cell_score: an ineligible challenger is never
            # better (tier -1), and a legacy holder (tier 0) yields to the first
            # program actually measured at this cell however large its global
            # fitness is.
            return self._cell_rank(program1, feature_key) > self._cell_rank(
                program2, feature_key
            )

        # Compare fitness (excluding feature dimensions)
        fitness1 = get_fitness_score(program1.metrics, self.config.feature_dimensions)
        fitness2 = get_fitness_score(program2.metrics, self.config.feature_dimensions)

        return fitness1 > fitness2

    def _update_archive(self, program: Program) -> None:
        """
        Update the archive of elite programs

        Args:
            program: Program to consider for archive
        """
        # If archive not full, add program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Clean up stale references and get valid archive programs
        valid_archive_programs = []
        stale_ids = []

        for pid in self.archive:
            if pid in self.programs:
                valid_archive_programs.append(self.programs[pid])
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.archive.discard(stale_id)
            logger.debug(f"Removing stale program {stale_id} from archive")

        # If archive is now not full after cleanup, just add the new program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Find worst program among valid programs
        if valid_archive_programs:
            worst_program = min(
                valid_archive_programs,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
            )

            # Replace if new program is better
            if self._is_better(program, worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:
            # No valid programs in archive, just add the new one
            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        # Compare with current best program (if it still exists)
        if self.best_program_id not in self.programs:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.programs[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:
        """
        Update the best program tracking for a specific island

        Args:
            program: Program to consider as the new best for the island
            island_idx: Island index
        """
        # Ensure island_idx is valid
        if island_idx >= len(self.island_best_programs):
            logger.warning(f"Invalid island index {island_idx}, skipping island best update")
            return

        # If island doesn't have a best program yet, this becomes the best
        current_island_best_id = self.island_best_programs[island_idx]
        if current_island_best_id is None:
            self.island_best_programs[island_idx] = program.id
            logger.debug(f"Set initial best program for island {island_idx} to {program.id}")
            return

        # Check if current best still exists
        if current_island_best_id not in self.programs:
            logger.warning(
                f"Island {island_idx} best program {current_island_best_id} no longer exists, updating to {program.id}"
            )
            self.island_best_programs[island_idx] = program.id
            return

        current_island_best = self.programs[current_island_best_id]

        # Update if the new program is better
        if self._is_better(program, current_island_best):
            old_id = current_island_best_id
            self.island_best_programs[island_idx] = program.id

            # Log the change
            if (
                "combined_score" in program.metrics
                and "combined_score" in current_island_best.metrics
            ):
                old_score = current_island_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id} "
                    f"(combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id}"
                )

    def _sample_parent(self) -> Program:
        """
        Sample a parent program from the current island for the next evolution step

        Returns:
            Parent program from current island
        """
        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            # This parent does not come from a cell, so the round serves no
            # corner and must not be told that it does (see cell_target_weight).
            self.bin_target = None
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample from any program (remaining probability)
            self.bin_target = None
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration (from current island)
        """
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            # If current island is empty, initialize with best program or random program
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Initialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                # Use any available program
                return next(iter(self.programs.values()))

        # Clean up stale references and sample from current island
        valid_programs = [pid for pid in current_island_programs if pid in self.programs]

        # Remove stale program IDs from island
        if len(valid_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(valid_programs)
            logger.debug(
                f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
            )
            for stale_id in stale_ids:
                self.islands[self.current_island].discard(stale_id)

        # If no valid programs after cleanup, reinitialize island
        if not valid_programs:
            logger.warning(
                f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
            )
            if self.best_program_id and self.best_program_id in self.programs:
                # Create a copy of best program for the empty island (don't reuse same ID)
                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Reinitialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                return next(iter(self.programs.values()))

        # Sample from valid programs
        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation: uniform over MAP-Elites cell elites
        (classic MAP-Elites parent rule - every occupied cell of the feature
        grid gets the same chance to parent, so frontier cells expand as fast
        as the global best).
        """
        # Clean up stale references in the feature map
        stale_keys = [key for key, pid in self.feature_map.items() if pid not in self.programs]
        for key in stale_keys:
            del self.feature_map[key]

        cell_elites = list(self.feature_map.values())
        if not cell_elites:
            # Fallback to exploration if the grid is empty
            self.bin_target = None
            return self._sample_exploration_parent()

        # Two-stage draw when per-cell pools are on: a CELL uniformly (one cell, one vote --
        # the coverage pressure a frontier search needs), then a program WITHIN that cell by
        # softmax over fitness. Uniform-over-elites is the special case pool size 1.
        try:
            _pool_k = int(os.environ.get("LAYOUT_CELL_POOL", "0") or 0)
            _temp = float(os.environ.get("LAYOUT_CELL_POOL_T", "0.2") or 0.2)
        except ValueError:
            _pool_k, _temp = 0, 0.2
        # The pools are written only in add(). Two paths populate feature_map WITHOUT add():
        # main.py's _policy_bin_rebuild under V2_POLICY_BINS=1, and load() on resume. In both
        # cases cell_pools is missing cells the grid already has, and drawing from it would
        # silently sample a stale -- possibly single-cell -- partition. Requiring the pools to
        # cover the grid makes those paths fall back to uniform-over-elites instead.
        _cp = getattr(self, "cell_pools", None)
        if _pool_k > 0 and _cp and len(_cp) >= len(self.feature_map):
            _keys = [k for k, v in self.cell_pools.items()
                     if any(pid in self.programs for pid in v)]
            # Bin-islands: parent comes from THIS slot's own bin (set by _submit_iteration).
            # Which cells may parent at all is a property of the ACTIVE dimension, not a
            # constant: see reproducing_cell_keys.
            if bin_islands_enabled():
                _keys = self.reproducing_cell_keys(_keys)
                _bt = getattr(self, "bin_target", None)
                if _bt is not None:
                    if str(_bt) in _keys:
                        _keys = [str(_bt)]
                    else:
                        # The target cell is EMPTY. Without this the draw fell
                        # through to the whole grid and the round quietly became
                        # an ordinary one, so empty cells were never attempted.
                        _nk = self._nearest_cell_key(_bt, _keys)
                        if _nk is not None:
                            _keys = [_nk]
            if _keys:
                # The drawn cell is recorded so the round can be told which
                # corner it serves (WEIGHT_ELITES "Why registration alone is not
                # enough"): a parent from cell w=9 whose overall best is w=13
                # otherwise gets asked only to "improve" and drifts back to 13.
                _key = random.choice(_keys)
                self.bin_target = _key
                _pool = [pid for pid in self.cell_pools[_key] if pid in self.programs]
                # SHAPE-UNIFORM PARENT. Group this cell's pool
                # by protograph shape, keep each shape's own best, rank the
                # shapes by it, and draw uniformly among the top K. Without
                # this the cell's elite - one shape - parents every
                # exploitation round in the cell and the other shapes of the
                # same rate band are never refined.
                try:
                    _shape_k = int(os.environ.get("LAYOUT_SHAPE_PARENT_K", "0") or 0)
                except ValueError:
                    _shape_k = 0
                if _shape_k > 0 and len(_pool) > 1:
                    _by_shape = {}
                    for _pid in _pool:
                        _sh = self._program_shape(self.programs[_pid]) or ("id", _pid)
                        _prev = _by_shape.get(_sh)
                        if _prev is None or self._cell_rank(self.programs[_pid], _key) > self._cell_rank(self.programs[_prev], _key):
                            _by_shape[_sh] = _pid
                    _ranked = sorted(_by_shape.values(),
                                     key=lambda pid: self._cell_rank(self.programs[pid], _key),
                                     reverse=True)[:_shape_k]
                    if _ranked:
                        _chosen = random.choice(_ranked)
                        logger.debug("shape-uniform parent: cell %s, %d shapes, drew %s",
                                     _key, len(_by_shape), _chosen[:8])
                        return self.programs[_chosen]
                # A program that is not eligible for this cell is not a parent
                # for it either: it would be drawn on a number measured at some
                # other weight. The pools are pruned on write too, so this only
                # catches entries a resume or a launcher rebuild put there.
                _elig = [pid for pid in _pool
                         if self._cell_eligible(self.programs[pid], _key)]
                _pool = _elig or _pool
                if _pool:
                    if len(_pool) == 1 or _temp <= 0:
                        return self.programs[_pool[0]]
                    # Softmax over THIS CELL's scores, not the global fitness: a
                    # policy whose winner is heavy is not the best parent for a
                    # light cell just because that winner is large.
                    _f = [self._cell_rank(self.programs[pid], _key)[1] for pid in _pool]
                    _m = max(_f)
                    _w = [math.exp((x - _m) / _temp) for x in _f]
                    _tot = sum(_w)
                    if _tot > 0:
                        _r = random.random() * _tot
                        for pid, wt in zip(_pool, _w):
                            _r -= wt
                            if _r <= 0:
                                return self.programs[pid]
                    return self.programs[_pool[0]]

        # Uniform over CELLS, not over elites: with multi-cell registration one
        # program can hold several cells, and drawing over the values would give
        # it one ticket per cell.
        #
        # Reached whenever the pools are off or do not cover the grid (a
        # _policy_bin_rebuild, a resume), which is a state the run can be in for
        # its whole length - so which cells may parent is decided here by the
        # same authority as in the pooled branch above. Drawing straight from
        # feature_map let the rate grid's out-of-window boundary bins reproduce,
        # the behaviour reproducing_cell_keys exists to remove.
        _cell_keys = [str(k) for k in self.feature_map.keys()]
        if bin_islands_enabled():
            _cell_keys = self.reproducing_cell_keys(_cell_keys)
            _bt = getattr(self, "bin_target", None)
            if _bt is not None:
                if str(_bt) in _cell_keys:
                    _cell_keys = [str(_bt)]
                else:
                    # Empty target cell: parent from the nearest occupied one
                    # (see _nearest_cell_key), not from the whole grid.
                    _nk = self._nearest_cell_key(_bt, _cell_keys)
                    if _nk is not None:
                        _cell_keys = [_nk]
        _cell_key = random.choice(_cell_keys)
        self.bin_target = _cell_key
        return self.programs[self.feature_map[_cell_key]]

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent from all programs
        """
        if not self.programs:
            raise ValueError("No programs available for sampling")

        # Sample randomly from all programs
        program_id = random.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step.

        For proper island-based evolution, inspirations are sampled ONLY from the
        current island, maintaining genetic isolation between islands.

        Args:
            parent: Parent program
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs from the current island
        """
        inspirations = []

        # Get the parent's island (should be current_island)
        parent_island = parent.metadata.get("island", self.current_island)

        # Get all programs from the current island
        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            logger.warning(f"Island {parent_island} has no programs for inspiration sampling")
            return []

        # Include the island's best program if available and different from parent.
        # LAYOUT_INSP_GLOBAL_BEST=1 substitutes the RUN's best: with two islands the island
        # best hides the actual champion from half the population.
        island_best_id = self.island_best_programs[parent_island]
        if os.environ.get("LAYOUT_INSP_GLOBAL_BEST", "0") == "1":
            _gb = getattr(self, "best_program_id", None)
            if _gb and _gb in self.programs and _gb != parent.id:
                island_best_id = _gb
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            island_best = self.programs[island_best_id]
            inspirations.append(island_best)
            logger.debug(
                f"Including island {parent_island} best program {island_best_id} in inspirations"
            )
        elif island_best_id is not None and island_best_id not in self.programs:
            # Clean up stale island best reference
            logger.warning(
                f"Island {parent_island} best program {island_best_id} no longer exists, clearing reference"
            )
            self.island_best_programs[parent_island] = None

        # Add top programs from the island as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self.get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)

        # Add diverse programs from within the island
        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)

            # Try to sample from different feature cells within the island
            feature_coords = self._calculate_feature_coords(parent)
            nearby_programs = []

            # Create a mapping of feature cells to island programs for efficient lookup
            island_feature_map = {}
            for prog_id in island_program_ids:
                if prog_id in self.programs:
                    prog = self.programs[prog_id]
                    prog_coords = self._calculate_feature_coords(prog)
                    cell_key = self._feature_coords_to_key(prog_coords)
                    island_feature_map[cell_key] = prog_id

            # Try to find programs from nearby feature cells within the island
            for _ in range(remaining_slots * 3):  # Try more times to find nearby programs
                # Perturb coordinates. Clamped per DIMENSION: a shared width
                # would propose coordinates one axis does not have (cells 9..19
                # of the nine-cell weight grid), which are never in the map, so
                # the neighbour search wasted its attempts on cells that cannot
                # exist.
                perturbed_coords = [
                    max(0, min(self._bins_for_dimension(dim) - 1, c + random.randint(-2, 2)))
                    for dim, c in zip(self.config.feature_dimensions or [], feature_coords)
                ]

                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in island_feature_map:
                    program_id = island_feature_map[cell_key]
                    if (
                        program_id != parent.id
                        and program_id not in [p.id for p in inspirations]
                        and program_id not in [p.id for p in nearby_programs]
                        and program_id in self.programs
                    ):
                        nearby_programs.append(self.programs[program_id])
                        if len(nearby_programs) >= remaining_slots:
                            break

            # If we still need more, add random programs from the island
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)

                # Get available programs from the island
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_island_ids = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded_ids and pid in self.programs
                ]

                if available_island_ids:
                    random_ids = random.sample(
                        available_island_ids, min(remaining, len(available_island_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        # Log island isolation info
        logger.debug(
            f"Sampled {len(inspirations)} inspirations from island {parent_island} "
            f"(island has {len(island_programs)} programs total)"
        )

        return inspirations[:n]

    def elite_ids(self) -> Set[str]:
        """Every program the archive is currently keeping as a cell elite.

        The global feature_map plus, when a launcher has installed them, the
        per-island feature maps (lps_graftgate/main.py builds
        db.island_feature_maps under V2_POLICY_BINS). Each of these is the best
        thing ever seen in some corner of the grid, which is the memory
        MAP-Elites exists to hold.
        """
        ids = {pid for pid in self.feature_map.values() if pid is not None}
        for island_map in getattr(self, "island_feature_maps", None) or []:
            ids.update(pid for pid in (island_map or {}).values() if pid is not None)
        return ids

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        """
        Enforce the population size limit by removing worst programs if needed

        Cell elites are protected. The pruner ranks by the GLOBAL fitness, and a
        light-weight elite is exactly a program whose global fitness is low: it
        was being deleted for the property that made it worth keeping, and the
        cell was emptied with it.

        Args:
            exclude_program_id: Program ID to never remove (e.g., newly added program)
        """
        if len(self.programs) <= self.config.population_size:
            return

        # Calculate how many programs to remove
        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )

        # Get programs sorted by fitness (worst first)
        all_programs = list(self.programs.values())

        # Sort by combined_score if available, otherwise by average metric (worst first)
        sorted_programs = sorted(
            all_programs,
            key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
        )

        # Remove worst programs, but never remove the best program or excluded program
        programs_to_remove = []
        protected_ids = {self.best_program_id, exclude_program_id} - {None}

        # ... and never remove a program that currently HOLDS a cell. The sort
        # above is the global fitness, and on the weight grid the light cells'
        # holders are the lowest-fitness programs there are (a w=6 elite at 14
        # against a heavy one at 80), so this pruner would delete exactly the
        # corner the dimension exists to preserve, taking the cell with it (see
        # the feature_map cleanup below).
        cell_holders = {pid for pid in self.elite_ids() if pid in self.programs}
        protected_ids |= cell_holders
        # v6c: the per-cell pools (LAYOUT_CELL_POOL) are archive memory too
        _pools = getattr(self, "cell_pools", None) or {}
        protected_ids |= {pid for ids in _pools.values() for pid in (ids or []) if pid in self.programs}

        for program in sorted_programs:
            if len(programs_to_remove) >= num_to_remove:
                break
            # Don't remove the best program or excluded program
            if program.id not in protected_ids:
                programs_to_remove.append(program)

        # The cap cannot be met without deleting elites. They are KEPT and the
        # population runs over the cap: dropping them would empty grid cells,
        # which is the one thing the archive exists to prevent, and it would
        # happen with no line in the log saying the run had stopped being a
        # MAP-Elites run. Logged at ERROR rather than raised, because a
        # multi-day run must not die on an arithmetic mistake in a config that
        # is fixable at the next checkpoint - but it is loud, and it names the
        # numbers needed to fix it.
        if len(programs_to_remove) < num_to_remove:
            kept = len(self.programs) - len(programs_to_remove)
            logger.error(
                "POPULATION CAP CANNOT BE MET WITHOUT DELETING CELL ELITES: "
                "population_size=%d, protected %d cell elite(s) + %d other, "
                "only %d of the %d programs due for removal are removable. "
                "Keeping every elite and running at %d programs (%d over the cap). "
                "Raise database.population_size above the number of occupied "
                "grid cells to clear this.",
                self.config.population_size,
                len(cell_holders),
                len(protected_ids - cell_holders),
                len(programs_to_remove),
                num_to_remove,
                kept,
                kept - self.config.population_size,
            )

        # Remove the selected programs
        for program in programs_to_remove:
            program_id = program.id

            # Remove from main programs dict
            if program_id in self.programs:
                del self.programs[program_id]

            # Remove from feature map
            keys_to_remove = []
            for key, pid in self.feature_map.items():
                if pid == program_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.feature_map[key]

            # Remove from islands
            for island in self.islands:
                island.discard(program_id)

            # Remove from archive
            self.archive.discard(program_id)

            logger.debug(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")

        # Clean up any stale island best program references after removal
        self._cleanup_stale_island_bests()

    # Island management methods
    def set_current_island(self, island_idx: int) -> None:
        """Set which island is currently being evolved"""
        self.current_island = island_idx % len(self.islands)
        logger.debug(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.debug(f"Island {idx} generation incremented to {self.island_generations[idx]}")

    def should_migrate(self) -> bool:
        """Check if migration should occur based on generation counters"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def migrate_programs(self) -> None:
        """
        Perform migration between islands

        This should be called periodically to share good solutions between islands
        """
        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            # Select top programs from this island for migration
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_programs:
                continue

            # Sort by fitness (using combined_score or average metrics)
            island_programs.sort(
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )

            # Select top programs for migration
            num_to_migrate = max(1, int(len(island_programs) * self.migration_rate))
            migrants = island_programs[:num_to_migrate]

            # Migrate to adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            for migrant in migrants:
                # Prevent re-migration of already migrated programs to avoid exponential duplication.
                # Analysis of actual evolution runs shows this causes severe issues:
                # - Program cb5d07f2 had 183 descendant copies by iteration 850
                # - Program 5645fbd2 had 31 descendant copies
                # - IDs grow exponentially: program_migrant_2_migrant_3_migrant_4_migrant_0...
                #
                # This is particularly problematic for OpenEvolve's MAP-Elites + Island hybrid architecture:
                # 1. All copies have identical code → same complexity/diversity/performance scores
                # 2. They all map to the SAME MAP-Elites cell → only 1 survives, rest discarded
                # 3. Wastes computation evaluating hundreds of identical programs
                # 4. Reduces actual diversity as islands fill with duplicates
                #
                # By preventing already-migrated programs from migrating again, we ensure:
                # - Each program migrates at most once per lineage
                # - True diversity is maintained between islands
                # - Computational resources aren't wasted on duplicates
                # - Aligns with MAP-Elites' one-program-per-cell principle
                if migrant.metadata.get("migrant", False):
                    continue

                for target_island in target_islands:
                    # Create a copy for migration (to avoid removing from source)
                    migrant_copy = Program(
                        id=f"{migrant.id}_migrant_{target_island}",
                        code=migrant.code,
                        language=migrant.language,
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        metrics=migrant.metrics.copy(),
                        metadata={**migrant.metadata, "island": target_island, "migrant": True},
                    )

                    # Add to target island
                    self.islands[target_island].add(migrant_copy.id)
                    self.programs[migrant_copy.id] = migrant_copy

                    # Update island-specific best program if migrant is better
                    self._update_island_best_program(migrant_copy, target_island)

                    # Log migration with MAP-Elites coordinates
                    feature_coords = self._calculate_feature_coords(migrant_copy)
                    coords_dict = {
                        self.config.feature_dimensions[j]: feature_coords[j]
                        for j in range(len(feature_coords))
                    }
                    logger.info(
                        "Program migrated to island %d at MAP-Elites coords: %s",
                        target_island,
                        coords_dict,
                    )

        # Update last migration generation
        self.last_migration_generation = max(self.island_generations)
        logger.info(f"Migration completed at generation {self.last_migration_generation}")

        # Validate migration results
        self._validate_migration_results()

    def _validate_migration_results(self) -> None:
        """
        Validate migration didn't create inconsistencies

        Checks that:
        1. Program island metadata matches actual island assignment
        2. No programs are assigned to multiple islands
        3. All island best programs exist and are in correct islands
        """
        seen_program_ids = set()

        for i, island in enumerate(self.islands):
            for program_id in island:
                # Check for duplicate assignments
                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)

                # Check program exists
                if program_id not in self.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue

                # Check metadata consistency
                program = self.programs[program_id]
                stored_island = program.metadata.get("island")
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: "
                        f"in island {i} but metadata says {stored_island}"
                    )

        # Validate island best programs
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                if best_id not in self.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best program references

        Cleans up references to programs that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.programs:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (program deleted)"
                    )
                    should_clear = True
                # Check if program is still in the island
                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale island best program references")

            # Recalculate best programs for islands that were cleared
            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and len(self.islands[i]) > 0:
                    # Find new best program for this island
                    island_programs = [
                        self.programs[pid] for pid in self.islands[i] if pid in self.programs
                    ]
                    if island_programs:
                        # Sort by fitness and update
                        best_program = max(
                            island_programs,
                            key=lambda p: p.metrics.get(
                                "combined_score", safe_numeric_average(p.metrics)
                            ),
                        )
                        self.island_best_programs[i] = best_program.id
                        logger.debug(f"Recalculated island {i} best program: {best_program.id}")

    def get_island_stats(self) -> List[dict]:
        """Get statistics for each island"""
        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]

            if island_programs:
                scores = [
                    get_fitness_score(p.metrics, self.config.feature_dimensions)
                    for p in island_programs
                ]

                best_score = max(scores) if scores else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        """Calculate diversity within an island (deterministic version)"""
        if len(programs) < 2:
            return 0.0

        total_diversity = 0
        comparisons = 0

        # Use deterministic sampling instead of random.sample() to ensure consistent results
        sample_size = min(5, len(programs))  # Reduced from 10 to 5

        # Sort programs by ID for deterministic ordering
        sorted_programs = sorted(programs, key=lambda p: p.id)

        # Take first N programs instead of random sampling
        sample_programs = sorted_programs[:sample_size]

        # Limit total comparisons for performance
        max_comparisons = 6  # Maximum comparisons to prevent long delays

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break

                # Use fast approximation instead of expensive edit distance
                diversity = self._fast_code_diversity(prog1.code, prog2.code)
                total_diversity += diversity
                comparisons += 1

            if comparisons >= max_comparisons:
                break

        return total_diversity / max(1, comparisons)

    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """
        Fast approximation of code diversity using simple metrics

        Returns diversity score (higher = more diverse)
        """
        if code1 == code2:
            return 0.0

        # Length difference (scaled to reasonable range)
        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)

        # Line count difference
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)

        # Simple character set difference
        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))

        # Combine metrics (scaled to match original edit distance range)
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5

        return diversity

    def _get_cached_diversity(self, program: Program) -> float:
        """
        Get diversity score for a program using cache and reference set

        Args:
            program: The program to calculate diversity for

        Returns:
            Diversity score (cached or newly computed)
        """
        code_hash = hash(program.code)

        # Check cache first
        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]

        # Update reference set if needed
        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()

        # Compute diversity against reference set
        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != program.code:  # Don't compare with itself
                diversity_scores.append(self._fast_code_diversity(program.code, ref_code))

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )

        # Cache the result with LRU eviction
        self._cache_diversity_value(code_hash, diversity)

        return diversity

    def _update_diversity_reference_set(self) -> None:
        """Update the reference set for diversity calculation"""
        if len(self.programs) == 0:
            return

        # Select diverse programs for reference set
        all_programs = list(self.programs.values())

        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs]
        else:
            # Select programs with maximum diversity
            selected = []
            remaining = all_programs.copy()

            # Start with a random program
            first_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))

            # Greedily add programs that maximize diversity to selected set
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1

                for i, candidate in enumerate(remaining):
                    # Calculate minimum diversity to selected programs
                    min_div = float("inf")
                    for selected_prog in selected:
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)

                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [p.code for p in selected]

        logger.debug(
            f"Updated diversity reference set with {len(self.diversity_reference_set)} programs"
        )

    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:
        """Cache a diversity value with LRU eviction"""
        # Check if cache is full
        if len(self.diversity_cache) >= self.diversity_cache_size:
            # Remove oldest entry
            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]

        # Add new entry
        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}

    def _invalidate_diversity_cache(self) -> None:
        """Invalidate the diversity cache when programs change significantly"""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
        logger.debug("Diversity cache invalidated")

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        """
        Update statistics for a feature dimension

        Args:
            feature_name: Name of the feature dimension
            value: New value to incorporate into stats
        """
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],  # Keep recent values for percentile calculation if needed
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Keep recent values for more sophisticated scaling methods
        stats["values"].append(value)
        if len(stats["values"]) > 1000:  # Limit memory usage
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        """
        Scale a feature value according to the configured scaling method

        Args:
            feature_name: Name of the feature dimension
            value: Raw feature value

        Returns:
            Scaled value in range [0, 1]
        """
        if feature_name not in self.feature_stats:
            # No stats yet, return normalized by a reasonable default
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5  # All values are the same

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))  # Ensure in [0, 1]

        elif self.feature_scaling_method == "percentile":
            # Use percentile ranking
            values = stats["values"]
            if not values:
                return 0.5

            # Count how many values are less than or equal to this value
            count = sum(1 for v in values if v <= value)
            percentile = count / len(values)
            return percentile

        else:
            # Default to min-max if unknown method
            return self._scale_feature_value_minmax(feature_name, value)

    def _scale_feature_value_minmax(self, feature_name: str, value: float) -> float:
        """Helper for min-max scaling"""
        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        if max_val == min_val:
            return 0.5

        scaled = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    def _serialize_feature_stats(self) -> Dict[str, Any]:
        """
        Serialize feature_stats for JSON storage

        Returns:
            Dictionary that can be JSON-serialized
        """
        serialized = {}
        for feature_name, stats in self.feature_stats.items():
            # Convert to JSON-serializable format
            serialized_stats = {}
            for key, value in stats.items():
                if key == "values":
                    # Limit size to prevent excessive memory usage
                    # Keep only the most recent 100 values for percentile calculations
                    if isinstance(value, list) and len(value) > 100:
                        serialized_stats[key] = value[-100:]
                    else:
                        serialized_stats[key] = value
                else:
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        serialized_stats[key] = value.item()
                    else:
                        serialized_stats[key] = value
            serialized[feature_name] = serialized_stats
        return serialized

    def _deserialize_feature_stats(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Deserialize feature_stats from loaded JSON

        Args:
            stats_dict: Dictionary loaded from JSON

        Returns:
            Properly formatted feature_stats dictionary
        """
        if not stats_dict:
            return {}

        deserialized = {}
        for feature_name, stats in stats_dict.items():
            if isinstance(stats, dict):
                # Ensure proper structure and types
                deserialized_stats = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
                deserialized[feature_name] = deserialized_stats
            else:
                logger.warning(
                    f"Skipping malformed feature_stats entry for '{feature_name}': {stats}"
                )

        return deserialized

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = (
                self.island_best_programs[island_idx]
                if island_idx < len(self.island_best_programs)
                else None
            )
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )

    # Artifact storage and retrieval methods

    def store_artifacts(self, program_id: str, artifacts: Dict[str, Union[str, bytes]]) -> None:
        """
        Store artifacts for a program

        Args:
            program_id: ID of the program
            artifacts: Dictionary of artifact name to content
        """
        if not artifacts:
            return

        program = self.get(program_id)
        if not program:
            logger.warning(f"Cannot store artifacts: program {program_id} not found")
            return

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"
        if not artifacts_enabled:
            logger.debug("Artifacts disabled, skipping storage")
            return

        # Split artifacts by size
        small_artifacts = {}
        large_artifacts = {}
        size_threshold = getattr(self.config, "artifact_size_threshold", 32 * 1024)  # 32KB default

        for key, value in artifacts.items():
            size = self._get_artifact_size(value)
            if size <= size_threshold:
                small_artifacts[key] = value
            else:
                large_artifacts[key] = value

        # Store small artifacts as JSON
        if small_artifacts:
            program.artifacts_json = json.dumps(small_artifacts, default=self._artifact_serializer)
            logger.debug(f"Stored {len(small_artifacts)} small artifacts for program {program_id}")

        # Store large artifacts to disk
        if large_artifacts:
            artifact_dir = self._create_artifact_dir(program_id)
            program.artifact_dir = artifact_dir
            for key, value in large_artifacts.items():
                self._write_artifact_file(artifact_dir, key, value)
            logger.debug(f"Stored {len(large_artifacts)} large artifacts for program {program_id}")

    def get_artifacts(self, program_id: str) -> Dict[str, Union[str, bytes]]:
        """
        Retrieve all artifacts for a program

        Args:
            program_id: ID of the program

        Returns:
            Dictionary of artifact name to content
        """
        program = self.get(program_id)
        if not program:
            return {}

        artifacts = {}

        # Load small artifacts from JSON
        if program.artifacts_json:
            try:
                small_artifacts = json.loads(program.artifacts_json)
                artifacts.update(small_artifacts)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode artifacts JSON for program {program_id}: {e}")

        # Load large artifacts from disk
        if program.artifact_dir and os.path.exists(program.artifact_dir):
            disk_artifacts = self._load_artifact_dir(program.artifact_dir)
            artifacts.update(disk_artifacts)

        return artifacts

    def _get_artifact_size(self, value: Union[str, bytes]) -> int:
        """Get size of an artifact value in bytes"""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def _artifact_serializer(self, obj):
        """JSON serializer for artifacts that handles bytes"""
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("utf-8")}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _artifact_deserializer(self, dct):
        """JSON deserializer for artifacts that handles bytes"""
        if "__bytes__" in dct:
            return base64.b64decode(dct["__bytes__"])
        return dct

    def _create_artifact_dir(self, program_id: str) -> str:
        """Create artifact directory for a program"""
        base_path = getattr(self.config, "artifacts_base_path", None)
        if not base_path:
            base_path = (
                os.path.join(self.config.db_path or ".", "artifacts")
                if self.config.db_path
                else "./artifacts"
            )

        artifact_dir = os.path.join(base_path, program_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _cleanup_old_artifacts(self, checkpoint_path: str) -> None:
        """
        Remove artifact directories older than the configured retention period.

        Args:
            checkpoint_path: The path of the current checkpoint being saved, which
                             contains the artifacts folder to be cleaned.
        """
        if not self.config.cleanup_old_artifacts:
            return

        artifacts_base_path = os.path.join(checkpoint_path, "artifacts")

        if not os.path.isdir(artifacts_base_path):
            return

        now = time.time()
        retention_seconds = self.config.artifact_retention_days * 24 * 60 * 60
        deleted_count = 0

        logger.debug(f"Starting artifact cleanup in {artifacts_base_path}...")

        for dirname in os.listdir(artifacts_base_path):
            dirpath = os.path.join(artifacts_base_path, dirname)
            if os.path.isdir(dirpath):
                try:
                    dir_mod_time = os.path.getmtime(dirpath)
                    if (now - dir_mod_time) > retention_seconds:
                        shutil.rmtree(dirpath)
                        deleted_count += 1
                        logger.debug(f"Removed old artifact directory: {dirpath}")
                except FileNotFoundError:
                    # Can happen in race conditions; ignore.
                    continue
                except Exception as e:
                    logger.error(f"Error removing artifact directory {dirpath}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old artifact directories.")

    def _write_artifact_file(self, artifact_dir: str, key: str, value: Union[str, bytes]) -> None:
        """Write an artifact to a file"""
        # Sanitize filename
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"

        file_path = os.path.join(artifact_dir, safe_key)

        try:
            if isinstance(value, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(value)
            elif isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                # Convert to string and write
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(value))
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")

    def _load_artifact_dir(self, artifact_dir: str) -> Dict[str, Union[str, bytes]]:
        """Load artifacts from a directory"""
        artifacts = {}

        try:
            for filename in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Try to read as text first
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:
                        # If text fails, read as binary
                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except Exception as e:
                        logger.warning(f"Failed to read artifact file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

        return artifacts

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """
        Log a prompt for a program.
        Only logs if self.config.log_prompts is True.

        Args:
        program_id: ID of the program to log the prompt for
        template_key: Key for the prompt template
        prompt: Prompts in the format {template_key: { 'system': str, 'user': str }}.
        responses: Optional list of responses to the prompt, if available.
        """

        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt
