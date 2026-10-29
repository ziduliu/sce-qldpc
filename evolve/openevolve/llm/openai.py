"""
OpenAI API interface for LLMs
"""

import asyncio
import os
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


LAST_REASONING_SUMMARY = {'text': ''}

class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

        # Token usage counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0
        # Price table: USD per 1M tokens (input, cached_input, output). Override via
        # env LLM_PRICE_IN / LLM_PRICE_CACHED / LLM_PRICE_OUT (per 1M).
        self._price_in = float(os.environ.get("LLM_PRICE_IN", "0.25"))
        self._price_cached = float(os.environ.get("LLM_PRICE_CACHED", "0.025"))
        self._price_out = float(os.environ.get("LLM_PRICE_OUT", "2.0"))

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # PROJECT PATCH: RANDOM_MUTATOR=1 bypasses the LLM entirely and applies
        # the leveled random mutation operator to the parent program instead.
        # Same prompts, same cards, same selection machinery - only the move
        # generator differs (control arm for the LLM-advantage comparison).
        import os as _os
        if _os.environ.get("RANDOM_MUTATOR") == "1":
            from openevolve.llm import random_mutator as _rm
            _text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            return _rm.respond(_text)
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.4, gpt-5.5
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = (
            self.api_base == "https://api.openai.com/v1"
            and model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)
        )

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            if "reasoning_effort" in kwargs:
                params["reasoning_effort"] = kwargs["reasoning_effort"]
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            if "reasoning_effort" in kwargs:
                params["reasoning_effort"] = kwargs["reasoning_effort"]

        # Env-gated reasoning effort (OPENAI_REASONING_EFFORT): lets a run turn
        # thinking on without config plumbing; absent -> provider default.
        _eff = os.environ.get("OPENAI_REASONING_EFFORT")
        if _eff and "reasoning_effort" not in params:
            params["reasoning_effort"] = _eff
        # Per-model override: OPENAI_EFFORT_MAP="modelA:high,modelB:medium"
        for _pair in os.environ.get("OPENAI_EFFORT_MAP", "").split(","):
            _m, _, _e = _pair.partition(":")
            if _m.strip() and _e.strip() and _m.strip() == str(self.model):
                params["reasoning_effort"] = _e.strip()
        # Per-instance config effort wins over map and env
        if getattr(self, "reasoning_effort", None):
            params["reasoning_effort"] = self.reasoning_effort

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        import os as _os
        if _os.environ.get("OPENAI_REASONING_SUMMARY", "0") == "1":
            try:
                return await loop.run_in_executor(
                    None, lambda: self._call_responses_api(params)
                )
            except Exception as _e:
                logging.getLogger(__name__).warning(
                    "responses API failed (%s); falling back to chat.completions", _e
                )
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.total_calls += 1
            _cached = getattr(getattr(response.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            _fresh_in = max(response.usage.prompt_tokens - _cached, 0)
            _call_cost = (_fresh_in * self._price_in + _cached * self._price_cached
                          + response.usage.completion_tokens * self._price_out) / 1e6
            self.total_cost_usd += _call_cost
            logger.info(
                f"COST this call: ${_call_cost:.5f} | Cumulative run cost: ${self.total_cost_usd:.4f}"
            )
            logger.info(
                f"Tokens this call: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens} "
                f"cached={getattr(getattr(response.usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0} "
                f"[{params['model']}] | "
                f"Cumulative: prompt={self.total_prompt_tokens}, "
                f"completion={self.total_completion_tokens}, "
                f"total={self.total_prompt_tokens + self.total_completion_tokens} "
                f"(calls={self.total_calls})"
            )
        return response.choices[0].message.content or ""

    def _call_responses_api(self, params: Dict[str, Any]) -> str:
        """Responses-API path: same generation, plus reasoning-summary capture.

        Cost/usage accounting mirrors the chat path. Every call appends one
        JSONL record (reasoning summary + answer head) for lineage analysis.
        """
        import os as _os, json as _json, time as _time
        logger = logging.getLogger(__name__)
        req = {
            "model": params["model"],
            "input": [
                {"role": m["role"], "content": m["content"]}
                for m in params["messages"]
            ],
            "reasoning": {
                "effort": params.get("reasoning_effort", "medium"),
                "summary": "detailed",
            },
        }
        if params.get("max_completion_tokens"):
            req["max_output_tokens"] = params["max_completion_tokens"]
        resp = self.client.responses.create(**req)
        content = getattr(resp, "output_text", "") or ""
        summaries = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "reasoning":
                for su in getattr(item, "summary", []) or []:
                    t = getattr(su, "text", "")
                    if t:
                        summaries.append(t)
        u = getattr(resp, "usage", None)
        if u is not None:
            _in = getattr(u, "input_tokens", 0) or 0
            _out = getattr(u, "output_tokens", 0) or 0
            _cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            _rsn = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            self.total_prompt_tokens += _in
            self.total_completion_tokens += _out
            self.total_calls += 1
            _call_cost = (
                max(_in - _cached, 0) * self._price_in
                + _cached * self._price_cached
                + _out * self._price_out
            ) / 1e6
            self.total_cost_usd += _call_cost
            logger.info(
                f"COST this call: ${_call_cost:.5f} | Cumulative run cost: ${self.total_cost_usd:.4f}"
            )
            logger.info(
                f"Tokens this call: prompt={_in}, completion={_out} cached={_cached} "
                f"reasoning={_rsn} [{params['model']}] (responses API) | "
                f"Cumulative: prompt={self.total_prompt_tokens}, "
                f"completion={self.total_completion_tokens}, "
                f"total={self.total_prompt_tokens + self.total_completion_tokens} "
                f"(calls={self.total_calls})"
            )
        log_path = _os.environ.get(
            "OPENAI_REASONING_LOG",
            "reasoning_%s.jsonl" % _os.environ.get("V2_ROUND_TAG", "run"),
        )
        try:
            with open(log_path, "a") as fh:
                fh.write(_json.dumps({
                    "ts": _time.time(),
                    "model": params["model"],
                    "effort": req["reasoning"]["effort"],
                    "summary": summaries,
                    "answer_head": content[:400],
                }) + "\n")
        except Exception as _e:
            logger.warning("reasoning log write failed: %s", _e)
        LAST_REASONING_SUMMARY["text"] = (" ".join(summaries))[:int(os.environ.get("OE_REASONING_CAP", "1000"))]
        return content
