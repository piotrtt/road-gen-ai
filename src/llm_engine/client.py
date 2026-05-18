"""
LLM client with two backends:

1. **DMGPT** (preferred when `DMGPT_PAT` is set): an OpenAI-compatible reverse
   proxy that fronts GPT, Gemini and Claude models with one credential. We call
   it through the official ``openai`` SDK because that's the only client guaranteed
   to support tool-calling against the proxy.
2. **litellm fallback** (when no DMGPT credentials are available): native
   provider routing as before.

The class shape is preserved (``query`` / ``query_structured``) so callers don't
care which backend is active.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import litellm

DMGPT_BASE_URL = "https://api.dmgpt.dm-drogeriemarkt.com/api/v1/openai"


class LLMClient:
    """
    Wrapper that abstracts interaction with different LLM providers.

    Routes through DMGPT (OpenAI-compatible) when ``DMGPT_PAT`` is set, otherwise
    falls back to litellm's provider-native routing.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Args:
            model_name: Model identifier (e.g. ``gpt-5.1``, ``gemini-2.5-pro``, ``claude-opus-4.6``).
            api_key: Override credential. If ``None``, the backend picks one up
                from the environment (``DMGPT_PAT`` for DMGPT, provider envs for litellm).
            temperature: Sampling temperature. If ``None`` no ``temperature`` field is sent
                and the provider default is used.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4")
        self.api_key = api_key
        self.temperature = temperature

        # Decide backend: DMGPT if PAT is available, otherwise litellm.
        self._dmgpt_pat = api_key or os.getenv("DMGPT_PAT")
        self._use_dmgpt = bool(self._dmgpt_pat)
        self._openai_client = None  # lazily constructed
        if self._use_dmgpt:
            # Import lazily so litellm-only environments don't need ``openai``.
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=DMGPT_BASE_URL,
                api_key=self._dmgpt_pat,
            )

    # ------------------------------------------------------------------ DMGPT
    def _dmgpt_completion(self, **kwargs):
        """Call DMGPT (OpenAI-compatible) and return the raw response."""
        # Strip out litellm-only fields.
        kwargs.pop("api_key", None)
        return self._openai_client.chat.completions.create(**kwargs)

    # ----------------------------------------------------------------- common
    def _temperature_kwargs(self) -> dict:
        return {"temperature": self.temperature} if self.temperature is not None else {}

    # =========================================================== plain query
    def query(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            if self._use_dmgpt:
                resp = self._dmgpt_completion(
                    model=self.model_name,
                    messages=messages,
                    **self._temperature_kwargs(),
                )
                return resp.choices[0].message.content
            else:
                resp = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    api_key=self.api_key,
                    **self._temperature_kwargs(),
                )
                return resp["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM Error] {e}")
            return f"Error querying LLM: {str(e)}"

    # ======================================================= structured (tools)
    def query_structured(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: str = "You are a helpful assistant.",
    ) -> Optional[Dict[str, Any]]:
        """
        Use function calling / tool use to get a structured output.

        Returns the parsed arguments of the first tool call, or ``None`` if no
        tool call was made / the call failed.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Force the model to call the first tool when one is provided.
        tool_choice: Any = "auto"
        if tools:
            tool_choice = {
                "type": "function",
                "function": {"name": tools[0]["function"]["name"]},
            }

        try:
            if self._use_dmgpt:
                resp = self._dmgpt_completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **self._temperature_kwargs(),
                )
                message = resp.choices[0].message
                if not getattr(message, "tool_calls", None):
                    return None
                tool_call = message.tool_calls[0]
                return json.loads(tool_call.function.arguments)
            else:
                resp = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    api_key=self.api_key,
                    **self._temperature_kwargs(),
                )
                message = resp["choices"][0]["message"]
                if not message.get("tool_calls"):
                    return None
                tool_call = message["tool_calls"][0]
                return json.loads(tool_call["function"]["arguments"])

        except Exception as e:
            print(f"[LLM Structured Error] {e}")
            return None

    # ====================================================== multi-turn tool call
    def tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        One round-trip on an arbitrary message history, forcing the model to
        invoke the named tool.

        ``messages`` is passed through verbatim — the caller controls the full
        conversation (system + user + prior assistant tool_call + tool result
        + new user turn, etc.). The response shape is:

            {"arguments": <parsed JSON dict>,
             "tool_call_id": <id string for use in a follow-up `tool` role message>,
             "raw_message": <the assistant message, suitable to be appended back
                             to ``messages`` for the next turn>}

        Returns ``None`` if the model didn't call the requested tool.
        """
        tool_choice = {"type": "function", "function": {"name": tool_name}}

        try:
            if self._use_dmgpt:
                resp = self._dmgpt_completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **self._temperature_kwargs(),
                )
                message = resp.choices[0].message
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls or tool_calls[0].function.name != tool_name:
                    return None
                tc = tool_calls[0]
                return {
                    "arguments": json.loads(tc.function.arguments),
                    "tool_call_id": tc.id,
                    "raw_message": {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            },
                        ],
                    },
                }
            else:
                resp = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    api_key=self.api_key,
                    **self._temperature_kwargs(),
                )
                message = resp["choices"][0]["message"]
                tool_calls = message.get("tool_calls") or []
                if not tool_calls or tool_calls[0]["function"]["name"] != tool_name:
                    return None
                tc = tool_calls[0]
                return {
                    "arguments": json.loads(tc["function"]["arguments"]),
                    "tool_call_id": tc["id"],
                    "raw_message": {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            },
                        ],
                    },
                }
        except Exception as e:
            print(f"[LLM tool_call Error] {e}")
            return None
