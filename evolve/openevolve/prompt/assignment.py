"""The assignment card, drawn BEFORE the prompt is assembled.

PROMPT_DESIGN.md sections 1 and 2. Two facts about this file:

1. The card is the FIRST section of the user message. It used to be appended
   to the end of the user template as a "{strategy_choice}" placeholder that
   openevolve filled afterwards, which put the only mandatory instruction of
   the round at character 147,566 of a 149,685-character message.
2. The card's LEVEL decides what else the prompt may contain, so the card has
   to be selected before assembly starts, not substituted into a finished
   template. A card is therefore parsed into (level, body) here and the
   builder branches on ``card.level``.

The deck still lives in the run config, under
``prompt.template_variations["strategy_choice"]`` - the key name is unchanged
so launchers, the forced-card mechanism and the level tagging in
lps_graftgate/evaluator.py keep working. What changed is that the sampler
consumes the deck itself instead of handing it to the stochastic template
substitution pass.

CARD FORMAT (one YAML string per card)::

    Assignment level: L3 (LEVEL_3_GROUP)
    Allowed fields: ...
    Frozen fields: ...
    Invalid if: ...

The "L<n>" is what the builder branches on. The parenthesised
``LEVEL_<n>_<NAME>`` tag is read by the prompt logger in
lps_graftgate/evaluator.py (regex ``LEVEL_[1-4]_[A-Z_]+``) to label each logged
call, so it stays in the text even though nothing in this module needs it.
"""

import os
import random
import re
from collections import namedtuple

# The heading the builder prints above every card. Tests assert that this
# string appears inside the first 500 characters of the user message.
ASSIGNMENT_HEADING = "# THIS ROUND - HIGHEST PRIORITY"

# Short restatement that may close the message: allowed fields only, never the
# whole card (PROMPT_DESIGN section 1).
ASSIGNMENT_TAIL_HEADING = "# END OF PROMPT - THE ASSIGNMENT AGAIN, IN ONE LINE"

_LEVEL_RE = re.compile(r"Assignment\s+level:\s*L([1-9])\b", re.IGNORECASE)
_ALLOWED_RE = re.compile(r"^\s*Allowed\s+fields:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# Levels that get a toolbox, and the template that holds it. L1 is absent on
# purpose: "no toolbox at all, only the parent's own failure evidence".
_TOOLBOX_TEMPLATE = {2: "toolbox_l2", 3: "toolbox_l3"}


class AssignmentCard(namedtuple("AssignmentCard", "level body")):
    """One drawn card: its level (int or None) and its text."""

    __slots__ = ()

    @property
    def allowed_fields(self):
        """The 'Allowed fields:' line, or "" when the card has none."""
        m = _ALLOWED_RE.search(self.body or "")
        return m.group(1).strip() if m else ""

    def header_block(self):
        """The card as it appears at the TOP of the user message.

        The NEXT_MOVE instruction is stripped when self-selection is off: it
        lives in the deck, so it would otherwise reach the control arm and tell
        that model to set a field nothing reads.
        """
        body = strip_self_selection_sentence((self.body or "").strip()).strip()
        return "%s\n%s" % (ASSIGNMENT_HEADING, body) if body else ""

    def tail_block(self):
        """The one-line restatement that may close the message."""
        allowed = self.allowed_fields
        if not allowed:
            return ""
        level = "L%d" % self.level if self.level else "this round"
        return "%s\nAssignment %s. Allowed fields: %s" % (
            ASSIGNMENT_TAIL_HEADING,
            level,
            allowed,
        )

    def toolbox_template(self):
        """Name of the toolbox template for this level, or None for no toolbox."""
        return _TOOLBOX_TEMPLATE.get(self.level)


def parse_card(text):
    """Parse one card string into an AssignmentCard.

    A card whose level cannot be read is returned with level None: it is still
    printed (an assignment the builder cannot classify is not an assignment
    that may be dropped), but it gets NO toolbox, which is the conservative
    end of section 2.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    m = _LEVEL_RE.search(text)
    level = int(m.group(1)) if m else None
    return AssignmentCard(level=level, body=text.strip())


def cards_from_config(prompt_config, key="strategy_choice"):
    """Every card in the deck, in config order."""
    variations = getattr(prompt_config, "template_variations", None) or {}
    deck = variations.get(key) or []
    out = []
    for text in deck:
        card = parse_card(text)
        if card is not None:
            out.append(card)
    return out


def select_card(prompt_config, key="strategy_choice", level=None, rng=None):
    """Draw the card for this round BEFORE the prompt is assembled.

    level -- force a specific assignment level (used by the tests and by any
             caller that wants a deterministic round); the draw is uniform over
             the whole deck when it is None, which is the production path.
    """
    deck = cards_from_config(prompt_config, key)
    if not deck:
        return None
    if level is not None:
        matching = [c for c in deck if c.level == level]
        if not matching:
            return None
        deck = matching
    chooser = rng if rng is not None else random
    card = chooser.choice(deck)
    _dealer_log(card, chooser)
    return card


def _dealer_log(card, chooser):
    """PROJECT DEBUG PROBE (env-gated): who draws the card, from what RNG state.

    Unchanged in behaviour from the probe that used to sit in
    PromptSampler._apply_template_variations; it moved here with the draw.
    """
    import os

    if os.environ.get("DEALER_LOG") != "1":
        return
    try:
        import hashlib

        state = hashlib.md5(repr(chooser.getstate()).encode()).hexdigest()[:8]
        with open("/tmp/dealer_probe.log", "a") as f:
            f.write(
                "pid=%d ppid=%d state=%s card=%s\n"
                % (
                    os.getpid(),
                    os.getppid(),
                    state,
                    (card.body or "")[:60].replace("\n", " "),
                )
            )
    except Exception:
        pass


# --------------------------------------------------------------------------
# Self-proposed rounds (V2_CARD_SELF=1)
#
# The deck is dealt uniformly at random by default and the model only executes.
# That is what makes the L3 card the mechanism behind the only advantage this
# project has measured: nothing else makes a model leave the family it is in.
# It also means the model is never asked whether it agrees, so "can it tell
# what it needs next" has never been measured either.
#
# Under V2_CARD_SELF a program declares, for its OWN children, which level it
# thinks the next round should be. The declaration travels with the program:
# when it is drawn as a parent, its NEXT_MOVE decides the level. It is a
# proposal, not a decision -- a quota still forces a level every Nth round, so
# a model that would sit in L1 forever cannot, and so all three levels keep
# producing samples whatever the models ask for.
#
# What it asked for and what it got are both logged, which is the point: a
# round that declared L3 and then gained is evidence about the model's own
# diagnosis, and a model that always declares L1 is evidence too.

_NEXT_MOVE_RE = re.compile(r"^\s*NEXT_MOVE\s*=\s*([123])\b", re.M)
# NEXT_WHY is written the way every other prose field in these programs is:
# sometimes one quoted string, sometimes several concatenated inside
# parentheses across lines. Grab the whole right-hand side up to the next
# top-level assignment and collect the string literals out of it, rather than
# assuming a quote follows the equals sign -- the first version did, and read
# every parenthesised reason as empty.
_NEXT_WHY_RE = re.compile(
    r"^\s*NEXT_WHY\s*=\s*(.*?)(?=\n[A-Za-z_]\w*\s*=|\n#|\ndef |\Z)", re.M | re.S)
_STR_LIT_RE = re.compile(r"(['\"])((?:\\.|(?!\1).)*)\1", re.S)


_SELF_SENTENCE_RE = re.compile(r"\s*Before you finish, set NEXT_MOVE.*?recorded\.", re.S)


def strip_self_selection_sentence(body):
    """Remove the NEXT_MOVE instruction from a card when the arm is off.

    The instruction lives in the deck, so it reaches the control arm too, where
    the declaration is parsed by nobody and changes nothing. Telling a model to
    set a field that decides nothing is a false statement in the one artefact
    that is supposed to be true (PROMPT_DESIGN section 3), and it also makes
    the control's prompt differ from the prompt every earlier run used, which
    is the thing a control exists not to do.
    """
    if self_selection_enabled():
        return body
    return _SELF_SENTENCE_RE.sub("", body or "")


def parse_next_move(code):
    """(level, reason) a program declares for its children, or (None, "").

    Read from the source rather than from the metrics because it is the
    model's own text and belongs to the genotype, not to the measurement. A
    program that declares nothing, or declares something outside 1..3, gets
    None and is dealt from the deck like any other round.
    """
    if not code:
        return None, ""
    m = _NEXT_MOVE_RE.search(code)
    if not m:
        return None, ""
    why = ""
    w = _NEXT_WHY_RE.search(code)
    if w:
        why = " ".join("".join(t[1] for t in _STR_LIT_RE.findall(w.group(1))).split())[:300]
    return int(m.group(1)), why


def self_selection_enabled():
    return os.environ.get("V2_CARD_SELF", "0") == "1"


def quota_level(round_index, levels=(1, 2, 3)):
    """The level round `round_index` is FORCED to, or None if it is free.

    Every V2_CARD_QUOTA-th round is forced and the forced level cycles, so
    each level keeps getting rounds no matter what the programs propose. 0
    switches the quota off entirely, which hands the schedule to the models --
    useful as its own arm, not as a default.
    """
    # The switch is checked HERE as well as in the caller. A quota that fires
    # when self-selection is off would force levels in the control arm, which
    # is the one thing this arm must not touch, and it would do it silently.
    if not self_selection_enabled():
        return None
    try:
        every = int(os.environ.get("V2_CARD_QUOTA", "3"))
    except ValueError:
        every = 3
    if every <= 0 or round_index <= 0 or round_index % every:
        return None
    return levels[((round_index // every) - 1) % len(levels)]
