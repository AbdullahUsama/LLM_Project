from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field


class GuardrailConfig(BaseModel):
    domain_keywords: list[str] = Field(
        default_factory=lambda: [
            "nust",
            "bank",
            "account",
            "card",
            "transfer",
            "mobile banking",
            "internet banking",
            "fee",
            "limit",
            "eligibility",
            "loan",
            "atm",
            "transaction",
        ]
    )
    out_of_scope_keywords: list[str] = Field(
        default_factory=lambda: [
            "weather",
            "movie",
            "music",
            "recipe",
            "football",
            "cricket",
            "politics",
            "javascript",
            "python code",
        ]
    )
    out_of_domain_reply: str = "I can only help with NUST Bank product and app questions."
    insufficient_context_reply: str = (
        "I don't have enough information in the provided knowledge base to answer that accurately."
    )
    min_context_chars: int = 40


default_guardrails = GuardrailConfig()


def is_clearly_out_of_scope(query: str, config: GuardrailConfig = default_guardrails) -> bool:
    q = query.lower()
    has_out_scope_signal = any(keyword in q for keyword in config.out_of_scope_keywords)
    has_bank_signal = any(keyword in q for keyword in config.domain_keywords)
    return has_out_scope_signal and not has_bank_signal


def has_sufficient_context(
    contexts: Iterable[str],
    config: GuardrailConfig = default_guardrails,
) -> bool:
    combined = " ".join(text.strip() for text in contexts if text)
    return len(combined) >= config.min_context_chars


def contains_unsupported_numbers(answer: str, context: str) -> bool:
    answer_numbers = set(re.findall(r"\\b\\d+(?:\\.\\d+)?%?\\b", answer))
    context_numbers = set(re.findall(r"\\b\\d+(?:\\.\\d+)?%?\\b", context))
    return bool(answer_numbers) and not answer_numbers.issubset(context_numbers)
