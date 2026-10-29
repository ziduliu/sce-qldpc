"""
Prompt templates for OpenEvolve
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution.
# {assignment} is FIRST: the card that governs the round must be inside the
# first 500 characters of the user message (PROMPT_DESIGN section 1). {toolbox}
# is whatever the card's level authorises, and is empty at L1 (section 2).
# Neither placeholder is load-bearing: PromptSampler prepends the card if a
# template is missing the slot, so a stale template file cannot silently drop
# the assignment.
DIFF_USER_TEMPLATE = """{assignment}{toolbox}# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Improve the program's performance on the specified metrics within the assignment above.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """{assignment}{toolbox}# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""

# Template for formatting a previous attempt.
# "Previous Attempts" is the current program's OWN LINEAGE (ancestors,
# chronological, direct parent last), so each entry states its relation and the
# iteration it was found at, and carries its EVOLVE block unless that block was
# already printed elsewhere in the prompt ({code_section} then holds a pointer).
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number} ({relation}, found at iteration {iteration})
- Changes: {changes}
- Metrics: {performance}
- Outcome: {outcome}
{code_section}"""

# Template for formatting a top program.
# {genotype} is a description of the program built from COMPUTED facts - its
# shape, the group order the size law forced, the support pattern of the built
# cells, the lightest witness the distance search found. It replaced a loop
# over the metrics that emitted "Performs well on <metric> (<value>)" for every
# key, which turned a 100%-failure placeholder into a stated strength
# (PROMPT_DESIGN section 5).
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
{genotype}
```{language}
{program_snippet}
```
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, {program_type})
{genotype}
```{language}
{program_snippet}
```
"""

# Toolbox injected when the assignment card is L2: the arithmetic a shape
# change has to satisfy, and nothing else (PROMPT_DESIGN section 2). Kept in
# sync with prompts/defaults/toolbox_l2.txt, which overrides it when present.
TOOLBOX_L2_TEMPLATE = """# TOOLBOX FOR THIS ASSIGNMENT LEVEL

SIZE LAW. A_SHAPE = (ra, ca) and B_SHAPE = (rb, cb) fix the factor
f = ca*cb + ra*rb, and the built code has n = f * q, where q is the order of
the group. The scaffold picks the LARGEST q whose n lands in the window stated
in the SCORING section and rejects the program outright if no q does.

ROW-WEIGHT ARITHMETIC. One HX row sums one A-row (ca cells) with one B-column
(rb cells); one HZ row sums one B-row (cb cells) with one A-column (ra cells).
A row's weight is the TOTAL number of group elements in those cells, and both
sums are gated against the cap in the SCORING section.

SHAPE COMPATIBILITY. Every cell of the new shape must be defined by SLOT 3:
CELL is called for side "A" at every (i, j) with i < ra and j < ca, and for
side "B" at every (i, j) with i < rb and j < cb.
"""

# Toolbox injected when the assignment card is L3: the group interface, the
# legality contract, and an explicit authorisation to create a new family. It
# names no candidate concept - describing the freedom is the point, listing
# answers is what anchors the search. Kept in sync with
# prompts/defaults/toolbox_l3.txt, which overrides it when present.
TOOLBOX_L3_TEMPLATE = """# TOOLBOX FOR THIS ASSIGNMENT LEVEL

AUTHORISATION. This round you may define a group that appears nowhere in this
prompt, nowhere in the archive and in no registry. Any finite group you can
write down as the interface below is admissible.

THE GROUP INTERFACE. CUSTOM_GROUP is a code STRING the scaffold evaluates. It
must define valid_q(q) -> bool (the orders your family exists at) and
custom_ops(q) -> dict with keys elements, ident, mul, inv, idx, red. Those
elements are what SLOT 3 draws from through G.

THE LEGALITY CONTRACT. The scaffold checks the identity, the inverses and
associativity before building and rejects a group that fails. n = (ca*cb +
ra*rb) * q must land in the window stated in the SCORING section for at least
one q with valid_q(q) true; the largest such q is used. An HX row covers
ca + rb cells and an HZ row ra + cb cells, counted in group elements, and both
totals are gated against the cap in the SCORING section.
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "toolbox_l2": TOOLBOX_L2_TEMPLATE,
    "toolbox_l3": TOOLBOX_L3_TEMPLATE,
}


class TemplateManager:
    """Manages templates with cascading override support"""

    def __init__(self, custom_template_dir: Optional[str] = None):
        # Get default template directory
        self.default_dir = Path(__file__).parent.parent / "prompts" / "defaults"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None

        # Load templates with cascading priority
        self.templates = {}
        self.fragments = {}

        # 0. The in-module DEFAULT_TEMPLATES are the floor. They used to be
        #    unreachable - the manager only ever read .txt files - so a template
        #    that existed as a constant but not as a file raised ValueError.
        #    Seeding them first means a missing prompts/defaults/*.txt degrades
        #    to the constant instead of taking the prompt down, and every .txt
        #    still overrides its constant below.
        self.templates.update(DEFAULT_TEMPLATES)

        # 1. Load defaults
        self._load_from_directory(self.default_dir)

        # 2. Override with custom templates (if provided)
        if self.custom_dir and self.custom_dir.exists():
            self._load_from_directory(self.custom_dir)

    def _load_from_directory(self, directory: Path) -> None:
        """Load all templates and fragments from a directory"""
        if not directory.exists():
            return

        # Load .txt templates
        for txt_file in directory.glob("*.txt"):
            template_name = txt_file.stem
            with open(txt_file, "r") as f:
                self.templates[template_name] = f.read()

        # Load fragments.json if exists. NOTE: fragments (fitness_improved,
        # fitness_declined, already_shown_ref, ...) live ONLY in
        # prompts/defaults/fragments.json - the DEFAULT_TEMPLATES dict above
        # holds .txt template bodies, not fragments. Edit the json, not this
        # file, when changing fragment wording.
        fragments_file = directory / "fragments.json"
        if fragments_file.exists():
            with open(fragments_file, "r") as f:
                loaded_fragments = json.load(f)
                self.fragments.update(loaded_fragments)

    def get_template(self, name: str) -> str:
        """Get a template by name"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def get_optional(self, name: str, default: str = "") -> str:
        """Get a template by name, or `default` when there is no such template.

        Used for the per-level toolboxes: a level with no toolbox (L1) and a
        level whose toolbox file is missing must both produce an empty section,
        not an exception.
        """
        return self.templates.get(name, default)

    def get_fragment(self, name: str, **kwargs) -> str:
        """Get and format a fragment"""
        if name not in self.fragments:
            return f"[Missing fragment: {name}]"
        try:
            return self.fragments[name].format(**kwargs)
        except KeyError as e:
            return f"[Fragment formatting error: {e}]"

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template

    def add_fragment(self, fragment_name: str, fragment: str) -> None:
        """Add or update a fragment"""
        self.fragments[fragment_name] = fragment
