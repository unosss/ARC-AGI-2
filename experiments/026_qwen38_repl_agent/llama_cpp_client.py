"""Explicit llama.cpp tool-choice compatibility for the Experiment 026 runner.

The acquired public client remains byte-for-byte unchanged.  llama.cpp accepts
string tool_choice values, unlike vLLM's named-tool object.  On a forced turn
we therefore expose only the requested tool and use ``"required"``.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "public_notebook/output_files/agent"
if not HARNESS.exists():
    HARNESS = HERE / "agent"
sys.path.insert(0, str(HARNESS))
from client import ChatClient, normalise


def build_payload(client, messages, tools=None, max_tokens=None, seed=None,
                  tool_choice=None, template_kwargs=None):
    payload = {
        "model": client.model,
        "messages": messages,
        "max_tokens": max_tokens or client.max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": client.thinking,
            **client.template_kwargs,
            **(template_kwargs or {}),
        },
        **client.sampling,
    }
    if seed is not None:
        payload["seed"] = seed
    if client.reasoning_effort:
        payload["reasoning_effort"] = client.reasoning_effort
    if client.thinking_token_budget:
        payload["thinking_token_budget"] = client.thinking_token_budget
    if tools:
        if tool_choice:
            selected = [
                tool for tool in tools
                if tool.get("function", {}).get("name") == tool_choice
            ]
            if len(selected) != 1:
                raise ValueError(f"forced tool {tool_choice!r} not found exactly once")
            payload["tools"] = selected
            payload["tool_choice"] = "required"
        else:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
    return payload


class LlamaCppChatClient(ChatClient):
    """ChatClient using llama.cpp's string-only forced-tool protocol."""

    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None,
             tool_choice=None, template_kwargs=None):
        payload = build_payload(
            self, messages, tools, max_tokens, seed, tool_choice, template_kwargs,
        )
        self.last_payload = payload
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
                return normalise(json.loads(response.read().decode()))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            overflow = "maximum context length" in detail or "context length" in detail
            if (
                exc.code == 400
                and not overflow
                and "thinking_token_budget" in payload
                and any(word in detail.lower() for word in ("thinking", "budget", "reasoning"))
            ):
                self.thinking_token_budget = None
                return self.chat(
                    messages, tools, max_tokens, seed, timeout, tool_choice, template_kwargs,
                )
            return {"error": f"HTTP {exc.code}: {detail}", "context_overflow": overflow}
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            return {
                "error": text[:300],
                "context_overflow": False,
                "timeout": isinstance(exc, TimeoutError) or "timed out" in text,
            }
