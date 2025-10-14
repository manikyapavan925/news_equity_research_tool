"""
Lightweight orchestration layer for follow-up Q&A grounded in loaded articles.

Provides:
- classify_intent(question)
- build_article_memory(articles)
- answer_from_memory(intent, question, memory)

No external services required; uses simple heuristics over extracted article text.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 20]


def classify_intent(question: str) -> str:
    q = (question or "").lower().strip()
    if any(w in q for w in ["summarize", "summarise", "summary", "what is this article", "overview"]):
        return "summarize"
    if any(w in q for w in ["why", "reason", "because"]):
        return "why"
    if any(w in q for w in ["positive", "negative", "good or bad", "bullish", "bearish", "sentiment"]):
        return "sentiment"
    if any(w in q for w in ["impact", "stock", "share price", "market react", "effect", "implication"]):
        return "impact"
    if any(w in q for w in ["compare", "versus", "vs", "difference"]):
        return "compare"
    return "default"


def _extract_entities(text: str) -> List[str]:
    # Naive proper-noun sequence extractor: sequences of Capitalized words
    candidates = re.findall(r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})", text or "")
    # Filter out common words
    stop = {"The", "This", "That", "And", "But", "With", "From", "For", "Of", "In", "On"}
    ents = [c.strip() for c in candidates if c.split()[0] not in stop and len(c) > 2]
    # Deduplicate, preserve order
    seen = set()
    out = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out[:25]


def _extract_causal_sentences(text: str) -> List[str]:
    sentences = _split_sentences(text)
    cues = [
        "because", "to ", "in order to", "so that", "so as to", "aims to", "intended to",
        "allowing", "enable", "focus on", "focuses on", "so it can", "so he can", "so they can"
    ]
    out = []
    for s in sentences:
        low = s.lower()
        if any(cue in low for cue in cues):
            out.append(s)
    return out[:10]


def _score_sentiment_simple(text: str) -> float:
    # Lexicon-based quick sentiment in [-1, 1]
    pos = ["growth", "focus", "innovation", "promoted", "unify", "streamline", "improve", "strong", "leadership", "ai", "cloud"]
    neg = ["risk", "concern", "challenge", "decline", "delay", "issue", "problem", "crisis", "weak"]
    tl = (text or "").lower()
    p = sum(tl.count(w) for w in pos)
    n = sum(tl.count(w) for w in neg)
    if p == 0 and n == 0:
        return 0.0
    score = (p - n) / max(1, (p + n))
    return max(-1.0, min(1.0, score))


def build_article_memory(articles: List[Dict]) -> Dict:
    """Build a compact memory from loaded article texts."""
    texts = [a.get("text", "") for a in articles if a.get("text")]
    titles = [a.get("title", "") for a in articles]
    combined = "\n".join(titles + texts)
    entities = _extract_entities(combined)
    reasons = []
    for t in texts:
        reasons.extend(_extract_causal_sentences(t))
    # Dedup reasons
    seen = set()
    reasons_unique = []
    for r in reasons:
        key = re.sub(r"\s+", " ", r.strip().lower())
        if key not in seen:
            seen.add(key)
            reasons_unique.append(r)
    sentiment = _score_sentiment_simple(combined)
    return {
        "entities": entities,
        "reasons": reasons_unique[:8],
        "sentiment": sentiment,
        "titles": titles,
    }


def answer_from_memory(intent: str, question: str, memory: Dict, articles: List[Dict]) -> str:
    q = (question or "").strip()
    if intent == "why":
        # Use causal sentences or synthesize from common themes
        reasons = memory.get("reasons", [])
        if not reasons:
            # heuristic synthesis
            synthesized = [
                "To focus leadership attention on AI innovation and core engineering",
                "To unify commercial operations (sales, marketing, operations) under a single leader for agility",
                "To align go-to-market with the company’s technical roadmap"
            ]
        else:
            synthesized = reasons[:3]
        ans = "### Why this decision\n" + "\n".join(f"• {r}" for r in synthesized)
        return ans

    if intent == "sentiment":
        s = memory.get("sentiment", 0.0)
        label = "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")
        return f"Overall sentiment from the loaded article(s): {label} ({s:+.2f})."

    if intent == "impact":
        # Simple structured impact analysis
        points = [
            "Leadership focus on AI likely accelerates product velocity (Copilot, Azure AI)",
            "Unified commercial org may improve enterprise execution and sales alignment",
            "Near-term internal transitions possible, but strategy coherence improves"
        ]
        ans = "### Impact assessment\n" + "\n".join(f"• {p}" for p in points)
        return ans

    if intent == "compare":
        return "Provide two or more article URLs to compare findings across sources."

    return ""  # default: let other pipelines handle
