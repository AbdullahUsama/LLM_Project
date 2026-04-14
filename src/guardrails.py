from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field


class GuardrailConfig(BaseModel):
    out_of_domain_reply: str = "I can only help with NUST Bank product and app questions."
    insufficient_context_reply: str = (
        "I don't have enough information in the provided knowledge base to answer that accurately."
    )
    prompt_injection_reply: str = (
        "I can help with NUST Bank questions, but I cannot follow instructions to ignore policies or system rules."
    )
    sensitive_output_reply: str = (
        "I cannot share sensitive information such as OTPs, PINs, CVV, passwords, or full account/card numbers."
    )
    min_context_chars: int = 40
    min_query_tokens: int = 2
    min_context_relevance: float = 0.12
    domain_hint_terms: list[str] = Field(
        default_factory=lambda: [
            "bank",
            "account",
            "transfer",
            "payment",
            "balance",
            "transaction",
            "card",
            "debit",
            "credit",
            "loan",
            "branch",
            "atm",
            "statement",
            "beneficiary",
            "kyc",
            "iban",
            "swift",
            "wallet",
            "cheque",
            "deposit",
            "withdraw",
            "app",
            "mobile banking",
        ]
    )
    prompt_injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore (all|any|previous|prior) instructions",
            r"disregard (all|any|previous|prior)",
            r"reveal (the )?(system|developer) prompt",
            r"show (me )?(hidden|internal) instructions",
            r"jailbreak|do anything now|bypass",
        ]
    )
    sensitive_data_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b(?:otp|pin|cvv|cvc|password|passcode)\b",
            r"\b\d{13,19}\b",
        ]
    )


default_guardrails = GuardrailConfig()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def context_relevance_score(query: str, contexts: Iterable[str]) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    context_tokens = _tokenize(" ".join(c for c in contexts if c))
    if not context_tokens:
        return 0.0

    overlap = len(q_tokens.intersection(context_tokens))
    return overlap / max(len(q_tokens), 1)


def is_likely_in_domain(
    query: str,
    contexts: Iterable[str],
    config: GuardrailConfig = default_guardrails,
) -> bool:
    q = query.lower().strip()
    q_tokens = _tokenize(q)

    if len(q_tokens) < config.min_query_tokens:
        return True

    if any(term in q for term in config.domain_hint_terms):
        return True

    return context_relevance_score(query, contexts) >= config.min_context_relevance


def is_clearly_out_of_scope(
    query: str,
    contexts: Iterable[str] | None = None,
    config: GuardrailConfig = default_guardrails,
) -> bool:
    if not query.strip():
        return False
    if contexts is None:
        return False
    return not is_likely_in_domain(query, contexts, config)


def has_sufficient_context(
    contexts: Iterable[str],
    config: GuardrailConfig = default_guardrails,
) -> bool:
    combined = " ".join(text.strip() for text in contexts if text)
    return len(combined) >= config.min_context_chars


def looks_like_prompt_injection(query: str, config: GuardrailConfig = default_guardrails) -> bool:
    q = query.lower().strip()
    return any(re.search(pattern, q) for pattern in config.prompt_injection_patterns)


def contains_sensitive_data(text: str, config: GuardrailConfig = default_guardrails) -> bool:
    t = text.lower()
    return any(re.search(pattern, t) for pattern in config.sensitive_data_patterns)


def contains_unsupported_numbers(answer: str, context: str) -> bool:
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
    return bool(answer_numbers) and not answer_numbers.issubset(context_numbers)
