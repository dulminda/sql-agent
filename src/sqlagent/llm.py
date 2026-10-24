"""Chat-model interface for the agent, with a Claude implementation.

The agent only needs one call: messages + tools in, an assistant turn out. Keeping that
interface narrow lets tests script turns deterministically (see ``tests/conftest.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# USD per million tokens: (input, output, cache read). Cache writes bill at 1.25x input.
# Keep in sync with https://www.anthropic.com/pricing
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0
) -> float:
    i, o, r = PRICING.get(model, (0.0, 0.0, 0.0))
    return (
        input_tokens * i + output_tokens * o + cache_read * r
        + cache_write * i * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000


class ModelError(RuntimeError):
    """The model call failed or returned something unusable."""


class ModelRefusal(ModelError):
    """The model declined the request (stop_reason == "refusal")."""


@dataclass
class Turn:
    content: list[Any]  # content blocks; appended to history unchanged
    stop_reason: str
    input_tokens: int  # uncached input
    output_tokens: int
    model: str
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens,
                        self.cache_read_tokens, self.cache_write_tokens)

    def tool_uses(self) -> list[Any]:
        return [b for b in self.content if b.type == "tool_use"]

    def text(self) -> str:
        return "".join(b.text for b in self.content if b.type == "text")


class ChatModel(Protocol):
    model: str

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


class ClaudeChat:
    """Claude Messages API. Server-side refusal fallback is on by default.

    Note there is no forced ``tool_choice``: current Claude models reject ``any``/``tool``.
    The prompt asks the model to finish with ``submit_answer``, and tools use
    ``strict: true`` so their arguments always match the schema.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        fallback: bool = True,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)
        self.model, self.effort = model, effort
        self.max_tokens, self.fallback = max_tokens, fallback

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # Cache the stable prefix (tools + system) across the turns of one task.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
        )
        try:
            if self.fallback:
                r = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                r = self._client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:
            raise ModelError(f"Claude API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise ModelError(f"could not reach the Claude API: {e}") from e

        if r.stop_reason == "refusal":
            category = getattr(r.stop_details, "category", None)
            raise ModelRefusal(f"model declined the request (category={category})")
        u = r.usage
        return Turn(
            list(r.content), r.stop_reason, u.input_tokens, u.output_tokens, r.model,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
